#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AGI_blueprint.py

A hyper-scalable, self-evolving grid-based GCN architecture for MNIST,
inspired by additional functionality proposed in AGI_blueprint.ipynb.  
This code retains all existing features from the current infinigraph_dev.py
and expands them to be even more general, comprehensive, and production-ready.

KEY ENHANCEMENTS:
  1. Parallel/Chunked Processing (Concurrency) Placeholder:
     - Demonstrates how different supernodes or time steps can be processed in parallel.
     - This is a placeholder; actual parallelization will depend on hardware and frameworks.

  2. Multi-Head Support:
     - Allows dynamically attaching multiple classifier heads to the same grid embeddings
       (e.g., classification + segmentation heads).
     - An example "attach_head" method in SupernodeGrid.

  3. More Robust Dynamic Expansion:
     - expand_grid() can now expand in X and Y simultaneously, relocates existing supernodes
       properly, and re-initializes the newly added sub-grid.

  4. TaskClassifierGrid:
     - Illustrates meta-learning or new-task detection.
     - Summarily triggers dynamic expansion and concurrency/parallel hints.

  5. Thorough Comments & Docstrings:
     - Every function is annotated with parameters and usage instructions.
     - Code is fully commented for clarity.

Usage:
  python AGI_blueprint.py

Sequence:
  1) Train the main SupernodeGrid on MNIST (with a linear classification head).
  2) Use the secondary TaskClassifierGrid to detect novel tasks, expand the main grid if indicated.
  3) Evaluate on test set.
  4) Illustrate concurrency placeholders, multi-head attachment, wide expansions, etc.

Author: The Development Team
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torchvision import datasets, transforms
from tqdm import tqdm
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

###############################################################################
# 1. Single supernode definition: a 3×3 grid of nodes connected by GCN layers.
###############################################################################
class Supernode(nn.Module):
    """
    The Supernode class models a 3×3 (9-node) mini-graph. Each node can have
    in_channels features, and after two GCN layers, features transform to
    out_channels dimension.

    We optionally incorporate neighbor and temporal features, and an
    arbitrary_module for custom transformations.

    Concurrency Potential:
      • Each Supernode forward pass is fairly independent, so these calls could be
        executed in parallel across multiple devices if needed (placeholder).
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 arbitrary_module: nn.Module = None):
        """
        :param in_channels: Number of input features per node.
        :param out_channels: Number of output features per node.
        :param arbitrary_module: Optional custom nn.Module to process node embeddings.
        """
        super().__init__()

        # GCN layers to encode node features: in_channels -> out_channels -> out_channels
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

        # Linear adapters for neighbor + temporal data
        self.neighbor_proj = nn.Linear(out_channels, out_channels)
        self.temporal_proj = nn.Linear(out_channels, out_channels)
        self.neighbor_input_adapter = nn.Linear(in_channels, out_channels, bias=False)
        self.temporal_input_adapter = nn.Linear(in_channels, out_channels, bias=False)

        # Arbitrary sub-module for specialized transformations (e.g. attention block)
        self.arbitrary_module = arbitrary_module

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self,
                data: Data,
                neighbor_features: torch.Tensor = None,
                prev_time_features: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for one Supernode:
          1) GCN layers to transform the node features.
          2) Optional user-defined module (arbitrary_module).
          3) Summation with neighbor + temporal features after dimension checks.

        :param data: A torch_geometric 'Data' object with:
            - x: Node features [9 x in_channels or out_channels]
            - edge_index: Graph connectivity
        :param neighbor_features: (Optional) aggregated features from neighbor supernodes
        :param prev_time_features: (Optional) features from the previous time step
        :return: Updated node features [9 x out_channels].
        """
        x, edge_index = data.x, data.edge_index

        # 1a. First GCN with ReLU
        x = F.relu(self.conv1(x, edge_index))
        # 1b. Second GCN
        x = self.conv2(x, edge_index)

        # 2. Optionally pass through arbitrary module
        if self.arbitrary_module is not None:
            x = self.arbitrary_module(x)

        # 3a. Summation with neighbor features
        if neighbor_features is not None:
            if neighbor_features.shape[1] == self.in_channels:
                x += self.neighbor_input_adapter(neighbor_features)
            else:
                x += self.neighbor_proj(neighbor_features)

        # 3b. Summation with temporal features
        if prev_time_features is not None:
            if prev_time_features.shape[1] == self.in_channels:
                x += self.temporal_input_adapter(prev_time_features)
            else:
                x += self.temporal_proj(prev_time_features)

        return x


###############################################################################
# 2. Helper function to build a single dense supernode graph (3×3 fully connected).
###############################################################################
def create_dense_supernode_graph(size: int = 3,
                                 feature_dim: int = 16) -> Data:
    """
    Creates a 3×3 = 9-node fully connected graph (without self-loops).
    Each node is randomly initialized with feature_dim features.

    :param size: grid dimension (3 => 3×3).
    :param feature_dim: number of features per node.
    :return: A torch_geometric.data.Data object containing:
        - x: node features -> shape [9, feature_dim]
        - edge_index: edges in COO format (fully connected except diagonal).
    """
    num_nodes = size * size  # 9
    x = torch.randn((num_nodes, feature_dim))
    # Create adjacency matrix with 1s except diagonal
    adj_matrix = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
    edge_index, _ = dense_to_sparse(adj_matrix)
    return Data(x=x, edge_index=edge_index)


###############################################################################
# 3. Main SupernodeGrid that can expand, run multi-time-step sequences, etc.
###############################################################################
class SupernodeGrid:
    """
    The main multi-dimensional (X, Y, Z, T) grid. Each cell is a 3×3 supernode.
    We apply neighbor + temporal logic, can run expansions via expand_grid(),
    and can attach multiple classification heads.

    Key Functions:
      • reinitialize_grid() -> reset all supernodes to [9 x in_channels].
      • run_full_sequence()  -> loop over time steps with process_time_step().
      • detect_new_task()    -> placeholder, can trigger expand_grid().
      • expand_grid()        -> dynamically add new supernodes in X or Y dimension.
      • attach_head()        -> add an additional classifier (or other) head to the grid.
    """
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 z_dim: int,
                 t_steps: int,
                 in_channels: int,
                 out_channels: int,
                 supernode_class=Supernode):
        """
        Initialize the grid with dimension [x_dim, y_dim, z_dim], each cell is
        a 3×3 supernode graph. The grid evolves for t_steps in time.

        :param x_dim, y_dim, z_dim: spatial dimensions of the grid.
        :param t_steps: # of time steps to simulate.
        :param in_channels: initial feature dimension per node (e.g. 784 for MNIST).
        :param out_channels: final feature dimension after GCN.
        :param supernode_class: class to use for supernodes (defaults to Supernode).
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.t_steps = t_steps
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Single shared Supernode model
        self.supernode_model = supernode_class(in_channels, out_channels)

        # Base Data template for each 3×3 supernode
        self.template_data = create_dense_supernode_graph(size=3, feature_dim=in_channels)

        # current_grid and next_grid hold Data objects for each (x,y,z,t)
        self.current_grid = {}
        self.next_grid = {}
        for t in range(self.t_steps):
            for z in range(self.z_dim):
                for y in range(self.y_dim):
                    for x in range(self.x_dim):
                        self.current_grid[(x, y, z, t)] = self.template_data.clone()
                        self.next_grid[(x, y, z, t)] = self.template_data.clone()

        # Placeholder for multiple heads (e.g., multiple classifiers)
        self.additional_heads = {}

    def attach_head(self, name: str, head_module: nn.Module):
        """
        Attaches an additional head (e.g., classification or segmentation)
        to the grid's dictionary. The code can later feed embeddings into these heads.

        :param name: A unique string for referencing this head.
        :param head_module: Any neural network module that expects the flattened
            supernode embeddings as input.
        """
        self.additional_heads[name] = head_module

    def get_neighbor_features(self, x, y, z, t) -> torch.Tensor:
        """
        Averages node features from the 6 possible neighbors in ±x, ±y, ±z.
        If no valid neighbors, returns None.
        """
        neighbor_coords = [(x - 1, y, z), (x + 1, y, z),
                           (x, y - 1, z), (x, y + 1, z),
                           (x, y, z - 1), (x, y, z + 1)]
        neighbors = []
        for nx, ny, nz in neighbor_coords:
            if 0 <= nx < self.x_dim and 0 <= ny < self.y_dim and 0 <= nz < self.z_dim:
                neighbors.append(self.current_grid[(nx, ny, nz, t)].x)

        if len(neighbors) == 0:
            return None
        else:
            # shape [#neighbors, 9, features] -> mean over neighbors -> [9, features]
            return torch.stack(neighbors).mean(dim=0)

    def get_temporal_features(self, x, y, z, t) -> torch.Tensor:
        """
        Retrieves the node features from the previous time step (t-1), if it exists.
        Returns None if t=0 or invalid.
        """
        if t <= 0:
            return None
        return self.current_grid[(x, y, z, t - 1)].x

    def process_time_step(self, t: int):
        """
        Processes a single time step t for all (x,y,z). This involves:
          1) Retrieving neighbor features,
          2) Retrieving temporal features,
          3) Passing everything into the shared Supernode model,
          4) Storing results in next_grid (at the same t).
        After the pass, this grid's state can be moved forward.

        Concurrency Placeholder:
          • The iteration over z,y,x can be parallelized if needed.
        """
        for z in range(self.z_dim):
            for y in range(self.y_dim):
                for x in range(self.x_dim):
                    current_data = self.current_grid[(x, y, z, t)]
                    neighbor_data = self.get_neighbor_features(x, y, z, t)
                    temporal_data = self.get_temporal_features(x, y, z, t)
                    updated_features = self.supernode_model(
                        current_data,
                        neighbor_features=neighbor_data,
                        prev_time_features=temporal_data
                    )
                    self.next_grid[(x, y, z, t)].x = updated_features.clone()

    def run_full_sequence(self):
        """
        Runs the entire time evolution from t=0..(t_steps-1).
        After each time step, next_grid becomes current_grid, so that
        supernode updates accumulate in time.
        """
        for t in range(self.t_steps):
            self.process_time_step(t)
            # Move next_grid to current_grid at the same t
            for z in range(self.z_dim):
                for y in range(self.y_dim):
                    for x in range(self.x_dim):
                        self.current_grid[(x, y, z, t)].x = self.next_grid[(x, y, z, t)].x.clone()

    def reinitialize_grid(self):
        """
        Re-initializes the entire grid (both current_grid and next_grid) to the
        original [9 x in_channels] shape. This is crucial when we re-insert new data
        (like a fresh MNIST sample) that must be processed from scratch.
        """
        for t in range(self.t_steps):
            for z in range(self.z_dim):
                for y in range(self.y_dim):
                    for x in range(self.x_dim):
                        self.current_grid[(x, y, z, t)] = self.template_data.clone()
                        self.next_grid[(x, y, z, t)] = self.template_data.clone()

    def get_final_embeddings(self) -> torch.Tensor:
        """
        Retrieves the embeddings from the last time step (t_steps - 1) across all (x,y,z).
        We concatenate them into a single [N_supernodes * 9, out_channels] tensor.

        :return: Aggregated node embeddings from the final time step.
        """
        final_ts = self.t_steps - 1
        outputs = []
        for z in range(self.z_dim):
            for y in range(self.y_dim):
                for x in range(self.x_dim):
                    outputs.append(self.current_grid[(x, y, z, final_ts)].x)
        return torch.cat(outputs, dim=0)

    def expand_grid(self, expand_x=0, expand_y=0):
        """
        Dynamically expands the grid in X and/or Y dimension. This is especially useful
        if the "TaskClassifierGrid" detects a novel/complex scenario.

        For example, expand_x=1 means we add 1 supernode along the X dimension.

        Implementation Outline:
          1) Create new dictionaries new_current_grid, new_next_grid with updated dimensions.
          2) Copy existing supernodes to their same coordinate in the new structure.
          3) Initialize newly added supernodes from the template_data clone.
          4) Update self.x_dim/y_dim accordingly.
        """
        new_x_dim = self.x_dim + expand_x
        new_y_dim = self.y_dim + expand_y
        if expand_x <= 0 and expand_y <= 0:
            print("No expansion requested. Doing nothing.")
            return

        # Build new dictionaries
        new_current_grid = {}
        new_next_grid = {}

        # Time steps remain the same
        for t in range(self.t_steps):
            for z in range(self.z_dim):
                for ny in range(new_y_dim):
                    for nx in range(new_x_dim):
                        # If within old range, copy old data
                        if nx < self.x_dim and ny < self.y_dim:
                            new_current_grid[(nx, ny, z, t)] = self.current_grid[(nx, ny, z, t)]
                            new_next_grid[(nx, ny, z, t)] = self.next_grid[(nx, ny, z, t)]
                        else:
                            # Otherwise, initialize fresh supernode
                            new_current_grid[(nx, ny, z, t)] = self.template_data.clone()
                            new_next_grid[(nx, ny, z, t)] = self.template_data.clone()

        # Update class state
        self.x_dim = new_x_dim
        self.y_dim = new_y_dim
        self.current_grid = new_current_grid
        self.next_grid = new_next_grid
        print(f"Grid expanded to x_dim={self.x_dim}, y_dim={self.y_dim}.")

###############################################################################
# 4. A minimal TaskClassifierGrid for meta-learning or new-task detection.
###############################################################################
class TaskClassifierGrid(nn.Module):
    """
    A simpler grid that tries to classify or detect new tasks. If it decides
    a new task is encountered, it signals the main grid to expand or adapt.

    This is a stub demonstration showing how a meta-learning layer can be integrated.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: e.g., 784 for MNIST.
        :param out_channels: e.g., 16 for smaller transformations.
        """
        super().__init__()
        # For demonstration, just do a small linear transformation
        self.linear = nn.Linear(in_channels, out_channels)
        self.threshold = 0.2  # naive threshold for "new task" detection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [1, in_channels], e.g. a single flattened MNIST image
        :return: a feature embedding [1, out_channels]
        """
        return self.linear(x)

    def detect_new_task(self, x: torch.Tensor) -> bool:
        """
        A placeholder method that infers whether 'x' is sufficiently "novel".
        If so, returns True -> triggers main grid expansion.

        :param x: [1, in_channels]
        :return: True if new/novel, else False
        """
        embedding = self.forward(x)  # shape [1, out_channels]
        measure = embedding.abs().mean().item()
        return (measure > self.threshold)


###############################################################################
# NEW: Advanced Transformer-based CLM head for text generation
###############################################################################
class AdvancedCLMHead(nn.Module):
    """
    A wrapper around a modern, more advanced LM (e.g., StarCoder, Llama2) for text generation.
    By default, we instantiate a large model from the HuggingFace Hub. In practice, you may
    choose a smaller or bigger model depending on hardware constraints.
    """
    def __init__(self, model_name="bigcode/starcoder"):
        """
        :param model_name: The Hugging Face model name to load for causal LM.
                          E.g., "bigcode/starcoder", "meta-llama/Llama-2-7b-hf", etc.
        """
        super().__init__()
        print(f"Loading advanced CLM model: {model_name}")
        self.lm_model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids, labels=None):
        """
        :param input_ids: tokens [batch_size, seq_len]
        :param labels: same shape as input_ids for LM loss
        :return: A standard transformers CausalLMOutputWithCrossAttentions containing loss, logits, etc.
        """
        outputs = self.lm_model(input_ids=input_ids, labels=labels)
        return outputs


###############################################################################
# UPDATED: function to train a CLM on text data (now advanced transformer)
###############################################################################
def train_on_advanced_clm(grid_model, lines_of_text, epochs=1, checkpoint_path=None):
    """
    Demonstrates how to use the grid to process text (tokenized) and
    train a more advanced LM for next-token prediction. Outputs perplexity.

    :param grid_model: The existing SupernodeGrid.
    :param lines_of_text: A list of raw text lines from e.g. a JSONL dataset.
    :param epochs: Number of passes over the entire text dataset.
    :param checkpoint_path: If set, load/save the TransformersHead state dict.
    """
    # Build/Load a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")
    tokenizer.pad_token = tokenizer.eos_token  # ensure there's a pad token

    # Create our advanced CLM head
    clm_head = AdvancedCLMHead(model_name="bigcode/starcoder")

    # If checkpoint exists, resume
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading CLM checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path)
        clm_head.load_state_dict(state["transformers_head"])

    # Optimizer for the LM head + grid
    optimizer = torch.optim.Adam(list(clm_head.parameters()) +
                              list(grid_model.supernode_model.parameters()),
                              lr=1e-4)

    # Minimal training loop
    for epoch in range(epochs):
        total_loss = 0.0
        total_count = 0

        # Shuffle lines in a basic way
        random.shuffle(lines_of_text)

        for line in lines_of_text:
            # Some minimal text => tokens
            tokens = tokenizer(line, return_tensors="pt", truncation=True, max_length=128)
            input_ids = tokens["input_ids"]  # shape [1, seq_len]

            # For next-token prediction, the labels are the same shift
            labels = input_ids.clone()

            # Reinitialize the grid if we want to incorporate the text embedding
            # We do a simple approach: Flatten the entire text embedding or random chunk
            grid_model.reinitialize_grid()

            # Possibly embed something into the supernodes (placeholder)
            # In a real scenario, we might feed word embeddings into each supernode, etc.
            # For simplicity, we skip that and just let clm_head handle text.

            optimizer.zero_grad()
            outputs = clm_head(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_count += 1

        avg_loss = total_loss / max(1, total_count)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        print(f"Epoch {epoch+1}/{epochs} => Loss: {avg_loss:.4f}, Perplexity: {ppl:.4f}")

        # Save checkpoint
        if checkpoint_path:
            torch.save({
                "transformers_head": clm_head.state_dict()
            }, checkpoint_path)
            print(f"CLM checkpoint saved to {checkpoint_path}")


###############################################################################
# 5. Main function: train, test, demonstrate expansions, concurrency placeholders.
###############################################################################
def main():
    """
    The main entry point:

    1) Build a main SupernodeGrid with 2×2 in X/Y, 1 in Z, t_steps=3, in=784, out=64.
    2) Insert first 4 MNIST images into the grid (t=0).
    3) Train the entire system (train_on_mnist).
    4) Evaluate on test set (test_on_mnist).
    5) Illustrate concurrency placeholder and multi-head usage.
    6) Use TaskClassifierGrid to detect new tasks and expand main grid if needed.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Main grid for actual data processing
    main_grid = SupernodeGrid(
        x_dim=2,
        y_dim=2,
        z_dim=1,
        t_steps=3,
        in_channels=28 * 28,
        out_channels=64
    )

    # Fill the first 4 supernodes with MNIST images
    for i, (img, label) in enumerate(mnist_train):
        if i >= 4:
            break
        flattened = img.view(1, 28 * 28)
        expanded = flattened.expand(9, -1)
        x = i % 2
        y = (i // 2) % 2
        # only one z=0, t=0 dimension
        main_grid.current_grid[(x, y, 0, 0)].x = expanded.clone()

    # Build a smaller grid to classify tasks (just concept demonstration)
    meta_grid = TaskClassifierGrid(in_channels=28 * 28, out_channels=16)

    # Demonstrate multi-head usage: Suppose we want a second classifier
    # for some other digit-based task:
    # (In practice, we'd attach a real nn.Module. We'll attach a small linear just as a placeholder.)
    alt_classifier = nn.Linear(main_grid.x_dim * main_grid.y_dim * 9 * main_grid.out_channels, 10)
    main_grid.attach_head("alt_classifier", alt_classifier)

    print("Starting full training on MNIST...")
    train_on_mnist(main_grid,
                   mnist_train,
                   epochs=2,  # reduce epochs for demonstration
                   learn_rate=1e-3,
                   checkpoint_path="main_checkpoint.pt")
    print("Training complete.\n")

    # Evaluate on entire test set
    test_on_mnist(main_grid, checkpoint_path="main_checkpoint.pt")

    # Show final embeddings dimension from the last time step
    main_grid.run_full_sequence()
    embeddings = main_grid.get_final_embeddings()
    print(f"Embeddings shape after final time step: {embeddings.shape}")

    # Example: concurrency placeholder
    # In an advanced setting, run_full_sequence could be parallelized, e.g.:
    # concurrent.futures / multiprocess / GPU parallel. (Placeholder, not fully implemented.)
    print("Concurrency placeholder: if we had enough hardware, run_full_sequence might be parallelized.\n")

    # Check if the next sample presents a new task:
    next_sample = next(iter(mnist_train))[0].view(1, 28 * 28)  # random one
    is_new_task = meta_grid.detect_new_task(next_sample)
    if is_new_task:
        print("Meta-grid detected a new task, expanding main grid by 1 in X dimension.")
        main_grid.expand_grid(expand_x=1)
    else:
        print("Meta-grid indicates no new task.")


###############################################################################
# 6. Comprehensive train-on-MNIST + checkpointing. (Main pipeline)
###############################################################################
def train_on_mnist(
    grid_model: SupernodeGrid,
    mnist_dataset: torch.utils.data.Dataset,
    epochs: int = 1,
    learn_rate: float = 1e-3,
    checkpoint_path: str = None
):
    """
    Trains the grid_model on MNIST images. Each image is flattened and placed
    at t=0 of the grid, we run run_full_sequence(), then classify with a linear head.

    :param grid_model: The main SupernodeGrid that processes data.
    :param mnist_dataset: MNIST train set.
    :param epochs: number of passes over the dataset.
    :param learn_rate: learning rate for Adam optimizer.
    :param checkpoint_path: if provided, load/save states for resuming.
    """
    # 1) Build a linear classification head that sees embeddings from all supernodes
    num_supernodes = grid_model.x_dim * grid_model.y_dim * grid_model.z_dim
    input_dim = num_supernodes * 9 * grid_model.out_channels  # 9 nodes per supernode
    classifier_head = nn.Linear(input_dim, 10)

    # 2) If checkpoint exists, resume
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Found checkpoint {checkpoint_path}. Resuming training from it.")
        chk = torch.load(checkpoint_path)
        grid_model.supernode_model.load_state_dict(chk["model"])
        classifier_head.load_state_dict(chk["classifier"])

    # 3) Setup optimizer, loss
    optimizer = torch.optim.Adam(
        list(grid_model.supernode_model.parameters()) + list(classifier_head.parameters()),
        lr=learn_rate
    )
    loss_fn = nn.CrossEntropyLoss()

    data_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=True)

    # 4) Train for multiple epochs
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        loop = tqdm(enumerate(data_loader), total=len(data_loader),
                    desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for i, (img, label) in loop:
            # Re-initialize the grid to shape [9 x in_channels]
            grid_model.reinitialize_grid()

            # Flatten MNIST [1, 28, 28] => [1, 784], then expand to [9, 784]
            flattened = img.view(1, 28 * 28).detach()
            expanded = flattened.expand(9, -1)

            # Place the expanded features in the supernodes at t=0
            for z in range(grid_model.z_dim):
                for y in range(grid_model.y_dim):
                    for x in range(grid_model.x_dim):
                        grid_model.current_grid[(x, y, z, 0)].x = expanded.clone()

            # Zero grad, run time steps
            optimizer.zero_grad()
            grid_model.run_full_sequence()

            # Flatten final embeddings for classification
            final_embs = grid_model.get_final_embeddings()  # [N*9, out_channels]
            flat_emb = final_embs.view(1, -1)  # shape [1, N*9*out_channels]

            logits = classifier_head(flat_emb)  # shape [1, 10]
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == label).sum().item()
            total_count += label.size(0)

            loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(total_correct / total_count):.4f}"
            })

        avg_loss = total_loss / len(data_loader)
        avg_acc = total_correct / total_count
        print(f"Epoch {epoch+1} complete. Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

        # Save checkpoint each epoch
        if checkpoint_path:
            torch.save({
                "model": grid_model.supernode_model.state_dict(),
                "classifier": classifier_head.state_dict()
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")


###############################################################################
# 7. Evaluate on the MNIST test set using the main grid + linear head.
###############################################################################
def test_on_mnist(grid_model: SupernodeGrid, checkpoint_path: str = None):
    """
    Evaluates the grid_model on the MNIST test set. We re-load the same
    classification head from the checkpoint if available. No gradient used.

    :param grid_model: The main SupernodeGrid.
    :param checkpoint_path: If provided, load from checkpoint before testing.
    """
    # Rebuild the same linear classifier
    num_supernodes = grid_model.x_dim * grid_model.y_dim * grid_model.z_dim
    input_dim = num_supernodes * 9 * grid_model.out_channels
    classifier_head = nn.Linear(input_dim, 10)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Test phase: loading checkpoint from {checkpoint_path}.")
        chk = torch.load(checkpoint_path)
        grid_model.supernode_model.load_state_dict(chk["model"])
        classifier_head.load_state_dict(chk["classifier"])
    else:
        print("No checkpoint found. Testing with current weights.")

    # Switch to eval mode (e.g. if batchnorm or dropout exist)
    grid_model.supernode_model.eval()
    classifier_head.eval()

    # MNIST test set
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Testing", leave=True):
            # Re-init grid
            grid_model.reinitialize_grid()

            # Flatten + expand
            flattened = img.view(1, 28 * 28)
            expanded = flattened.expand(9, -1)

            # Place in supernodes at t=0
            for z in range(grid_model.z_dim):
                for y in range(grid_model.y_dim):
                    for x in range(grid_model.x_dim):
                        grid_model.current_grid[(x, y, z, 0)].x = expanded.clone()

            # Forward pass
            grid_model.run_full_sequence()

            final_embs = grid_model.get_final_embeddings()
            flat_emb = final_embs.view(1, -1)
            logits = classifier_head(flat_emb)
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    print(f"Test Accuracy on entire MNIST test set: {correct/total:.4f}")


###############################################################################
# 8. If run directly, execute main().
###############################################################################
if __name__ == "__main__":
    main()
