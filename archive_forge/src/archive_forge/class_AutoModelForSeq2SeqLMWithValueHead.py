import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .modeling_base import PreTrainedModelWrapper
class AutoModelForSeq2SeqLMWithValueHead(PreTrainedModelWrapper):
    """
    A seq2seq model with a value head in addition to the language model head.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained` and `push_to_hub` and also provides some additional
    functionalities such as `generate`.

    Args:
        pretrained_model (`transformers.PreTrainedModel`):
            The model to wrap. It should be a causal language model such as GPT2.
            or any model mapped inside the `AutoModelForSeq2SeqLM` class.
        kwargs:
            Additional keyword arguments passed along to the `ValueHead` class.
    """
    transformers_parent_class = AutoModelForSeq2SeqLM
    lm_head_namings = ['lm_head', 'embed_out', 'output_projection']
    supported_args = ('summary_dropout_prob', 'v_head_initializer_range', 'v_head_init_strategy')

    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)
        self.is_encoder_decoder = True
        if not self._has_lm_head():
            raise ValueError('The model does not have a language model head, please use a model that has one.')
        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
        self._init_weights(**v_head_kwargs)

    def _has_lm_head(self):
        for name, _module in self.pretrained_model.named_modules():
            if any((attribute in name for attribute in self.lm_head_namings)):
                return True
        return False

    def post_init(self, state_dict):
        """
        We add the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        for k in list(state_dict.keys()):
            if 'v_head.' in k:
                state_dict[k.replace('v_head.', '')] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict
        if hasattr(self.pretrained_model, 'hf_device_map'):
            if 'cpu' in self.pretrained_model.hf_device_map.values() or 'disk' in self.pretrained_model.hf_device_map.values():
                raise ValueError('The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models.')
            for name, module in self.pretrained_model.named_modules():
                if any((attribute in name for attribute in self.lm_head_namings)):
                    lm_head_device = module.weight.device
                    break
            self.v_head = self.v_head.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                """
                A hook that sets the device of the output of the model to the device of the first
                parameter of the model.

                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output
            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}
        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f'v_head.{k}'] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        self.pretrained_model.v_head = self.v_head
        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def _init_weights(self, **kwargs):
        """
        We initialize the weights of the value head.
        """
        initializer_range = kwargs.pop('v_head_initializer_range', 0.2)
        init_strategy = kwargs.pop('v_head_init_strategy', None)
        if init_strategy is None:
            pass
        elif init_strategy == 'normal':
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, **kwargs):
        kwargs['past_key_values'] = past_key_values
        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == 'PREFIX_TUNING':
            kwargs.pop('past_key_values')
        base_model_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
        last_hidden_state = base_model_output.decoder_hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss
        value = self.v_head(last_hidden_state).squeeze(-1)
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()
        return (lm_logits, loss, value)

    def generate(self, *args, **kwargs):
        """
        We call `generate` on the wrapped model.
        """
        return self.pretrained_model.generate(*args, **kwargs)