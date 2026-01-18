# Module Header
"""
RWKV Language Model Inference Script
====================================
This script provides an interface to generate text using the RWKV language model.
It demonstrates loading a pre-trained model and generating text based on a given context.

References:
- RWKV Language Model GitHub: https://github.com/BlinkDL/RWKV-LM
"""

# Imports
import numpy as np
import types
from typing import Optional, Tuple
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

# Set numpy print options for better readability
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# Tokenizer initialization
tokenizer = Tokenizer.from_file("20B_tokenizer.json")

# Model and generation parameters
args = types.SimpleNamespace()
args.MODEL_NAME = (
    "/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066"
)
args.n_layer = 24
args.n_embd = 1024

# Generation context and parameters
context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
NUM_TRIALS = 3
LENGTH_PER_TRIAL = 100
TEMPERATURE = 1.0
TOP_P = 0.85

########################################################################################################


class RWKV_RNN(torch.jit.ScriptModule):
    """
    A PyTorch ScriptModule for the RWKV RNN model for text generation.

    This class encapsulates the model architecture, loading pre-trained weights,
    and performing forward passes with state management for text generation.

    Attributes:
        args (SimpleNamespace): Configuration arguments including model path and dimensions.
        w (SimpleNamespace): A namespace containing the model's weights organized by layer.
    """

    def __init__(self, args: types.SimpleNamespace):
        """
        Initializes the RWKV_RNN model with pre-trained weights.

        Args:
            args (SimpleNamespace): Configuration arguments including model path and dimensions.
        """
        super().__init__()
        self.args = args
        self.eval()  # Set the module to inference mode.

        # Load pre-trained weights and adjust them as necessary.
        weights = torch.load(args.MODEL_NAME + ".pth", map_location="cpu")
        for key in weights.keys():
            if ".time_" in key:
                weights[key] = weights[key].squeeze()
            if ".time_decay" in key:
                weights[key] = -torch.exp(
                    weights[key].float()
                )  # Apply decay transformation.
            else:
                weights[key] = weights[key].float()  # Ensure float32 type.

        # Organize weights into a structured namespace for easier access.
        self.w = types.SimpleNamespace()
        self.w.blocks = {}
        for key in weights:
            parts = key.split(".")
            attribute_name = parts.pop()
            current_namespace = self.w
            for part in parts:
                if part.isdigit():
                    part = int(part)
                    if part not in current_namespace:
                        current_namespace[part] = types.SimpleNamespace()
                    current_namespace = current_namespace[part]
                else:
                    if not hasattr(current_namespace, part):
                        setattr(current_namespace, part, types.SimpleNamespace())
                    current_namespace = getattr(current_namespace, part)
            setattr(current_namespace, attribute_name, weights[key])

    def layer_norm(self, x: torch.Tensor, w: types.SimpleNamespace) -> torch.Tensor:
        """
        Applies layer normalization on the input tensor.

        Args:
            x (torch.Tensor): The input tensor to normalize.
            w (SimpleNamespace): The namespace containing weight and bias for normalization.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @torch.jit.script_method
    def channel_mixing(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        i: int,
        time_mix_k: torch.Tensor,
        time_mix_r: torch.Tensor,
        kw: torch.Tensor,
        vw: torch.Tensor,
        rw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs the channel mixing operation as part of the model's forward pass.

        Args:
            x (torch.Tensor): The input tensor.
            state (torch.Tensor): The current state tensor.
            i (int): The layer index.
            time_mix_k (torch.Tensor): Time mixing tensor for key.
            time_mix_r (torch.Tensor): Time mixing tensor for receptance.
            kw (torch.Tensor): Key weights.
            vw (torch.Tensor): Value weights.
            rw (torch.Tensor): Receptance weights.

        Returns:
            torch.Tensor: The result of the channel mixing operation.
        """
        xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
        state[5 * i + 0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))
        return r * (vw @ k)

    @torch.jit.script_method
    def time_mixing(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        i: int,
        time_mix_k: torch.Tensor,
        time_mix_v: torch.Tensor,
        time_mix_r: torch.Tensor,
        time_first: torch.Tensor,
        time_decay: torch.Tensor,
        kw: torch.Tensor,
        vw: torch.Tensor,
        rw: torch.Tensor,
        ow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs the time mixing operation as part of the model's forward pass.

        Args:
            x (torch.Tensor): The input tensor.
            state (torch.Tensor): The current state tensor.
            i (int): The layer index.
            time_mix_k (torch.Tensor): Time mixing tensor for key.
            time_mix_v (torch.Tensor): Time mixing tensor for value.
            time_mix_r (torch.Tensor): Time mixing tensor for receptance.
            time_first (torch.Tensor): Initial time tensor.
            time_decay (torch.Tensor): Time decay tensor.
            kw (torch.Tensor): Key weights.
            vw (torch.Tensor): Value weights.
            rw (torch.Tensor): Receptance weights.
            ow (torch.Tensor): Output weights.

        Returns:
            torch.Tensor: The result of the time mixing operation.
        """
        xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)
        state[5 * i + 1] = x
        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv

        aa = state[5 * i + 2]
        bb = state[5 * i + 3]
        pp = state[5 * i + 4]
        ww = time_first + k
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b
        ww = pp + time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        state[5 * i + 2] = e1 * aa + e2 * v
        state[5 * i + 3] = e1 * bb + e2
        state[5 * i + 4] = qq
        return ow @ (r * wkv)

    def forward(
        self, token: int, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RWKV RNN model.

        Args:
            token (int): The current token id to process.
            state (Optional[torch.Tensor]): The current state tensor. If None, initializes a new state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output tensor and the updated state tensor.
        """
        with torch.no_grad():
            # Initialize state if not provided
            if state is None:
                state = torch.zeros(self.args.n_layer * 5, self.args.n_embd)
                for i in range(self.args.n_layer):
                    state[5 * i + 4] = (
                        -1e30
                    )  # Initialize with -infinity for proper handling in time mixing

            # Embedding and initial layer normalization
            x = self.w.emb.weight[token]
            x = self.layer_norm(x, self.w.blocks[0].ln0)

            # Sequentially process through all layers
            for i in range(self.args.n_layer):
                att = self.w.blocks[i].att
                x = x + self.time_mixing(
                    self.layer_norm(x, self.w.blocks[i].ln1),
                    state,
                    i,
                    att.time_mix_k,
                    att.time_mix_v,
                    att.time_mix_r,
                    att.time_first,
                    att.time_decay,
                    att.key.weight,
                    att.value.weight,
                    att.receptance.weight,
                    att.output.weight,
                )
                ffn = self.w.blocks[i].ffn
                x = x + self.channel_mixing(
                    self.layer_norm(x, self.w.blocks[i].ln2),
                    state,
                    i,
                    ffn.time_mix_k,
                    ffn.time_mix_r,
                    ffn.key.weight,
                    ffn.value.weight,
                    ffn.receptance.weight,
                )

            # Final layer normalization and linear transformation
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state


##########################################################################################################


def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out


########################################################################################################

print(f"\nUsing CPU. Loading {args.MODEL_NAME} ...")
model = RWKV_RNN(args)

print(f"\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)")
init_state = None
for token in tokenizer.encode(context).ids:
    init_out, init_state = model.forward(token, init_state)

for TRIAL in range(NUM_TRIALS):
    print(f"\n\n--[ Trial {TRIAL} ]-----------------", context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out.clone(), init_state.clone()
    for i in range(LENGTH_PER_TRIAL):
        token = sample_logits(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        tmp = tokenizer.decode(all_tokens[out_last:])
        if "\ufffd" not in tmp:  # only print when we have a valid utf-8 string
            print(tmp, end="", flush=True)
            out_last = i + 1
        out, state = model.forward(token, state)
print("\n")
