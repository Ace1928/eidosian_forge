import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torchvision import datasets, transforms
from tqdm import tqdm  # NEW: for real-time progress bars
import os  # for checking if checkpoint file exists

###############################################################################
# 1. Single supernode definition: a 3×3 grid of nodes connected by GCN layers.
###############################################################################
class Supernode(nn.Module):
    """
    The Supernode class models a 3×3 grid of nodes (9 nodes total). Each node
    can have 'in_channels' features, and after applying two GCN layers, the
    node features typically transform to 'out_channels' dimension.

    We also support integrating features from neighbors (spatial) and from
    previous time steps (temporal). To handle possible dimension mismatches,
    we introduce linear adapters that automatically convert neighbor and
    temporal features if they differ from the current dimension.
    """
    def __init__(self, in_channels, out_channels, arbitrary_module=None):
        """
        Constructor for the Supernode:
        - Initializes two GCN layers for processing.
        - Initializes neighbor and temporal adapters to handle dimension mismatches.
        - Optionally attaches 'arbitrary_module' for entirely custom per-node computation.
        """
        super(Supernode, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.neighbor_proj = nn.Linear(out_channels, out_channels)
        self.temporal_proj = nn.Linear(out_channels, out_channels)
        self.neighbor_input_adapter = nn.Linear(in_channels, out_channels, bias=False)
        self.temporal_input_adapter = nn.Linear(in_channels, out_channels, bias=False)

        # Store references for dimension checks
        self.in_channels = in_channels
        self.out_channels = out_channels

        # NEW: Arbitrary sub-module that can be any nn.Module, e.g., a mini language model
        # For example, you could pass a large transformer here; each node's features
        # might become tokens or prompts. This ensures maximum extensibility.
        self.arbitrary_module = arbitrary_module

    def forward(self, data, neighbor_features=None, prev_time_features=None):
        """
        Forward pass of the Supernode model:
        - Applies two GCN layers to the node features.
        - Optionally calls 'arbitrary_module' if provided.
        - Adapts and adds neighbor (spatial) and temporal (previous time) features.
        """
        # Extract node features (x) and edge connectivity (edge_index) from the data object
        x, edge_index = data.x, data.edge_index

        # 1. Apply the first GCN layer, followed by a ReLU activation
        x = F.relu(self.conv1(x, edge_index))

        # 2. Apply the second GCN layer (no activation here, but it's optional)
        x = self.conv2(x, edge_index)

        # NEW: Optionally allow transforming x via an arbitrary module
        # This can encapsulate any architecture. Example usage:
        if self.arbitrary_module is not None:
            x = self.arbitrary_module(x)

        # 3. If neighbor features exist, adapt them to match x's dimension and then add
        if neighbor_features is not None:
            # If the neighbor dimension is still in_channels, use neighbor_input_adapter
            if neighbor_features.shape[1] == self.in_channels:
                x = x + self.neighbor_input_adapter(neighbor_features)
            else:
                # Otherwise, assume neighbor features are already out_channels
                x = x + self.neighbor_proj(neighbor_features)

        # 4. If temporal features exist, do the same adaptive approach
        if prev_time_features is not None:
            # If the temporal dimension matches in_channels, adapt it to out_channels
            if prev_time_features.shape[1] == self.in_channels:
                x = x + self.temporal_input_adapter(prev_time_features)
            else:
                # Otherwise, assume it is already out_channels
                x = x + self.temporal_proj(prev_time_features)

        # 5. Return the final node features for this supernode
        return x


###############################################################################
# 2. Function to build a single dense supernode graph (3×3 fully connected).
###############################################################################
def create_dense_supernode_graph(size=3, feature_dim=16):
    """
    Creates a 3×3 (9-node) fully connected graph (fully connected except self-loops).
    Randomly initializes the node features to demonstrate usage.

    :param size: Size of the grid in one dimension (3 means a 3×3 = 9-node supernode).
    :param feature_dim: Number of features per node (e.g., 16).
    :return: A torch_geometric Data object with:
             - x: node features [9 x feature_dim]
             - edge_index: edges in COO format, describing a fully connected 9-node graph
    """
    # Compute the total number of nodes
    num_nodes = size * size  # e.g., 3 * 3 = 9 for a 3×3 grid

    # Randomly initialize the node features as [num_nodes x feature_dim], e.g. [9 x 16]
    x = torch.randn((num_nodes, feature_dim))

    # Create a 9×9 adjacency matrix with ones everywhere except on the diagonal
    adj_matrix = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)

    # Convert from a dense adjacency matrix to a sparse edge_index representation
    edge_index, _ = dense_to_sparse(adj_matrix)

    # Return a graph Data object containing features (x) and edges (edge_index)
    return Data(x=x, edge_index=edge_index)


###############################################################################
# 3. Assembling grids of supernodes across X, Y, Z, and T dimensions.
###############################################################################
class SupernodeGrid:
    """
    The SupernodeGrid class represents a multi-dimensional grid of Supernodes,
    arranged in X, Y, Z, and time (T). Each Supernode is a 3×3 mini-graph
    internally.

    Typical usage pattern:
    1) Create a SupernodeGrid with specific x, y, z dimensions, time steps (t_steps),
       and in/out channels specifying the feature sizes in the supernode.
    2) Optionally assign node features (like images) into self.current_grid at t=0.
    3) Call run_full_sequence() to sequentially process each time step, allowing
       neighbor + temporal communication.
    4) Retrieve final node embeddings with get_final_embeddings().
    """
    def __init__(self, x_dim, y_dim, z_dim, t_steps, in_channels, out_channels):
        """
        Constructor for SupernodeGrid:
        - Creates a single shared Supernode model with the given in_channels/out_channels.
        - Creates a template supernode Data object (3×3) with in_channels features.
        - Initializes two dictionaries, current_grid and next_grid, each storing
          a Data object for every position (x, y, z) at each time step.

        :param x_dim: Number of supernodes along the X dimension.
        :param y_dim: Number of supernodes along the Y dimension.
        :param z_dim: Number of supernodes along the Z dimension.
        :param t_steps: Number of time steps to process sequentially.
        :param in_channels: Size of each node's initial feature vector (e.g. 784 for MNIST).
        :param out_channels: Size of the node feature vector after GCN transformations.
        """
        # Store spatial dimensions and number of time steps
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.t_steps = t_steps

        # Store the input and output channel dimensions for reference
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create one shared Supernode model, used identically for all supernodes
        self.supernode_model = Supernode(in_channels, out_channels)

        # Create the "template_data" from which all individual supernode Data objects are cloned
        self.template_data = create_dense_supernode_graph(size=3, feature_dim=in_channels)

        # Dictionaries to hold the Data objects for every coordinate/time:
        # current_grid is for the current time step
        # next_grid is for the next time step
        self.current_grid = {}
        self.next_grid = {}

        # Initialize these dictionaries so each position/time has a separate Data clone
        for t in range(t_steps):
            for z in range(z_dim):
                for y in range(y_dim):
                    for x in range(x_dim):
                        # Store a cloned template for both current and next grids at (x, y, z, t)
                        self.current_grid[(x, y, z, t)] = self.template_data.clone()
                        self.next_grid[(x, y, z, t)] = self.template_data.clone()

    def get_neighbor_features(self, x, y, z, t):
        """
        Gathers neighbor features for the supernode at (x, y, z) within the same time step t.
        By default, we average the node features of valid neighbors. The neighbor
        coordinates are offset by ±1 in X, Y, or Z.

        :param x: X coordinate of the current supernode.
        :param y: Y coordinate of the current supernode.
        :param z: Z coordinate of the current supernode.
        :param t: Current time step.
        :return: A [9 x feature_dim] tensor containing the average of neighbor node features,
                 or None if no neighbors exist (i.e., out of bounds).
        """
        neighbors = []  # A list to collect neighbor node feature tensors

        # Define 6 possible directions in 3D space: ±x, ±y, ±z
        directions = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1)
        ]

        # Check each direction for valid neighbors
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            # Validate that (nx, ny, nz) is within the grid boundaries
            if (0 <= nx < self.x_dim and
                0 <= ny < self.y_dim and
                0 <= nz < self.z_dim):
                # If valid, retrieve and append that neighbor's node features
                neighbors.append(self.current_grid[(nx, ny, nz, t)].x)

        # If there are no valid neighbors, return None
        if len(neighbors) == 0:
            return None
        else:
            # Otherwise, stack all neighbor feature tensors along dimension 0 and take the mean
            # The shape will remain [9 x feature_dim], but it's the average across neighbors
            return torch.stack(neighbors).mean(dim=0)

    def get_temporal_features(self, x, y, z, t):
        """
        Retrieves the features of the same (x, y, z) supernode but at the previous time step (t-1),
        if it exists (i.e., if t > 0).

        :param x: X coordinate of the supernode.
        :param y: Y coordinate of the supernode.
        :param z: Z coordinate of the supernode.
        :param t: Current time step.
        :return: A [9 x feature_dim] tensor if t > 0, otherwise None.
        """
        if t > 0:
            # Return the features from the previous time step
            return self.current_grid[(x, y, z, t - 1)].x
        else:
            # If this is the first time step, there's no previous step
            return None

    def process_time_step(self, t):
        """
        This method processes a single time step 't' for all supernodes in the grid.
        For each supernode, we:
        1) Retrieve its neighbor features (same t) via get_neighbor_features.
        2) Retrieve its temporal features (t-1) via get_temporal_features.
        3) Pass them all into self.supernode_model along with the current node features
           to obtain updated_features.
        4) Store updated_features in next_grid for time step t (to be swapped later).

        :param t: The time step index to process.
        """
        # Iterate over all possible z, y, x positions
        for z in range(self.z_dim):
            for y in range(self.y_dim):
                for x in range(self.x_dim):
                    # Construct a key for current_grid index
                    data_key = (x, y, z, t)

                    # Retrieve the graph Data object for this position/time
                    data = self.current_grid[data_key]

                    # Gather neighbor features for the same time step
                    neighbor_feats = self.get_neighbor_features(x, y, z, t)

                    # Gather temporal features from the previous time step
                    temporal_feats = self.get_temporal_features(x, y, z, t)

                    # Forward pass through the shared supernode model
                    updated_features = self.supernode_model(
                        data,
                        neighbor_features=neighbor_feats,
                        prev_time_features=temporal_feats
                    )

                    # Store these updated features in next_grid so we can swap later
                    self.next_grid[data_key].x = updated_features

        # After processing all supernodes at this time step, swap current_grid and next_grid
        # so that the newly updated features become the "current" features for the next iteration.
        self.current_grid, self.next_grid = self.next_grid, self.current_grid

    def run_full_sequence(self):
        """
        Runs the entire T-step process. For each t in range(t_steps), we call process_time_step(t).
        By the end, self.current_grid holds all the final (time = t_steps - 1) features.
        """
        for t in range(self.t_steps):
            self.process_time_step(t)

    def get_final_embeddings(self):
        """
        Gathers and concatenates the final node embeddings (after all time steps) from
        the entire (x, y, z) grid at time t_steps - 1.

        :return: A single tensor of shape [(x_dim * y_dim * z_dim * 9) x out_channels] if all
                 supernodes end up with out_channels dimension.
        """
        # The final time step is t_steps - 1
        t = self.t_steps - 1

        # Collect each supernode's node features in a list
        outputs = []
        for z in range(self.z_dim):
            for y in range(self.y_dim):
                for x in range(self.x_dim):
                    # Extract node features at the final time step
                    outputs.append(self.current_grid[(x, y, z, t)].x)

        # Concatenate all node features along dimension 0 (stack them vertically)
        return torch.cat(outputs, dim=0)

    def reinitialize_grid(self):
        """
        Re-initialize the entire grid (current_grid and next_grid) to the original
        template state. This ensures that each supernode's Data object starts
        with shape [9 x in_channels]. This prevents shape mismatch that arises
        if we repeatedly re-use a grid whose nodes have already become [9 x out_channels].

        We simply clone the template_data for every coordinate (x, y, z) and time t.
        Called before processing each training sample to ensure GCNConv
        (in_channels->out_channels) has the correct input dimension.
        """
        for t in range(self.t_steps):
            for z in range(self.z_dim):
                for y in range(self.y_dim):
                    for x in range(self.x_dim):
                        self.current_grid[(x, y, z, t)] = self.template_data.clone()
                        self.next_grid[(x, y, z, t)] = self.template_data.clone()


###############################################################################
# 4. Example usage with MNIST to demonstrate how data could flow into the grid.
###############################################################################
def main():
    """
    Main function that:
    1) Demonstrates building a grid model for MNIST images,
    2) Fills it with the first 4 images,
    3) Runs the time evolution,
    4) Prints the final embedding shape.
    (No training by default in this function.)
    """
    # Transform that converts PIL images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # MNIST dataset with the above transform, downloaded locally if needed
    mnist_dataset = datasets.MNIST(
        root='./data',      # Directory to store MNIST data
        train=True,         # True -> training dataset
        download=True,      # Download on first run if not already present
        transform=transform # Apply the transform to each image
    )

    # We create a grid of 2×2 in X and Y, 1 in Z, and we process for 3 time steps
    # Each supernode is a 3×3 mini-graph with initially 784 features (28x28).
    # After going through the GCN layers, each node should end up with 64 features.
    grid_model = SupernodeGrid(
        x_dim=2,
        y_dim=2,
        z_dim=1,
        t_steps=3,
        in_channels=28 * 28,   # 784 dimension from a flattened 28x28 MNIST image
        out_channels=64        # Output dimension after the GCN
    )

    # We fill only the first 4 supernodes (2×2×1 = 4) at time step t=0, each
    # with a flattened MNIST image expanded to 9 nodes
    for i, (img, label) in enumerate(mnist_dataset):
        if i < 4:
            # Flatten the 28x28 image into shape [1, 784]
            flattened_img = img.view(1, 28 * 28)

            # Expand or "tile" this 1×784 row so that all 9 nodes in the supernode
            # share the same image features (shape becomes 9×784).
            flattened_img_9nodes = flattened_img.expand(9, -1)

            # Determine grid location (x, y, z=0, t=0) for the i-th MNIST sample
            x = i % 2
            y = (i // 2) % 2
            z = 0
            t = 0

            # Assign these features to the data object in the current_grid
            grid_model.current_grid[(x, y, z, t)].x = flattened_img_9nodes
        else:
            # After placing 4 MNIST samples, stop
            break

    # --------------------------------------------------------------------------
    # NEW: Perform a comprehensive training run before final demonstration
    # --------------------------------------------------------------------------
    print("Starting comprehensive training on MNIST:")
    # Train for 5 epochs on the *entire* MNIST train set, saving and resuming from a checkpoint
    train_on_mnist(grid_model, mnist_dataset, epochs=5, learn_rate=1e-3,
                   checkpoint_path="model_checkpoint.pt")
    print("Training complete.\n")

    # After training, run a full test pass on the MNIST test set:
    test_on_mnist(grid_model, checkpoint_path="model_checkpoint.pt")

    # Run the time evolution from t=0 up to t=2 just as a final demonstration
    grid_model.run_full_sequence()
    embeddings = grid_model.get_final_embeddings()
    print("Final embeddings shape:", embeddings.shape)
    # We expect [36, 64] for (2×2×1) × 9 nodes × 64 features.


###############################################################################
# 5. Minimal addition: Demonstration of how to train the entire system on MNIST.
###############################################################################
def train_on_mnist(grid_model, mnist_dataset, epochs=1, learn_rate=1e-3, checkpoint_path=None):
    """
    A simple example of end-to-end training:
    - We repeatedly take MNIST samples, flatten and put them in the supernode at t=0.
    - We run the full time sequence, retrieve embeddings, then classify with a linear head.
    - We compute cross-entropy vs the digit labels and backprop through the entire system.
    - We save a checkpoint at the end of each epoch to allow resuming.

    :param grid_model: An instance of SupernodeGrid.
    :param mnist_dataset: The MNIST dataset (train set).
    :param epochs: Number of passes over the entire train dataset.
    :param learn_rate: Learning rate for optimizer.
    :param checkpoint_path: If not None, loads/saves model+classifier weights here.
    """
    import torch

    # (A) Classifier head: dimension is (#supernodes*9*out_channels) -> 10 classes
    num_supernodes = grid_model.x_dim * grid_model.y_dim * grid_model.z_dim
    input_dim = num_supernodes * 9 * grid_model.out_channels
    classifier_head = nn.Linear(input_dim, 10)

    # If checkpoint_path exists, resume from it (model + classifier)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Found checkpoint '{checkpoint_path}'. Resuming...")
        checkpoint = torch.load(checkpoint_path)
        grid_model.supernode_model.load_state_dict(checkpoint["supernode_model_state"])
        classifier_head.load_state_dict(checkpoint["classifier_head_state"])
    else:
        print("No checkpoint found. Training from scratch.")

    # (B) Gather all parameters: supernode GCNs + classifier head
    optimizer = torch.optim.Adam(
        list(grid_model.supernode_model.parameters()) + list(classifier_head.parameters()),
        lr=learn_rate
    )
    loss_fn = nn.CrossEntropyLoss()

    # (C) Use a DataLoader for iteration
    from torch.utils.data import DataLoader
    data_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        loop = tqdm(enumerate(data_loader), total=len(data_loader),
                    desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for i, (img, label) in loop:
            # Re-initialize the grid each time so that each supernode is back to [9 x in_channels].
            grid_model.reinitialize_grid()

            flattened_img = img.view(1, 28 * 28).detach()
            expanded_img = flattened_img.expand(9, -1)
            for z in range(grid_model.z_dim):
                for y in range(grid_model.y_dim):
                    for x in range(grid_model.x_dim):
                        grid_model.current_grid[(x, y, z, 0)].x = expanded_img.clone()

            optimizer.zero_grad()
            grid_model.run_full_sequence()
            final_embeddings = grid_model.get_final_embeddings()
            flat_emb = final_embeddings.view(1, -1)
            logits = classifier_head(flat_emb)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred_label = logits.argmax(dim=1)
            total_correct += (pred_label == label).sum().item()
            total_samples += label.size(0)

            # Update the progress bar
            loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(total_correct / (total_samples)): .4f}"
            })

        avg_loss = total_loss / len(data_loader)
        avg_acc = total_correct / total_samples
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")

        # Save checkpoint at the end of each epoch
        if checkpoint_path is not None:
            torch.save({
                "supernode_model_state": grid_model.supernode_model.state_dict(),
                "classifier_head_state": classifier_head.state_dict()
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

def test_on_mnist(grid_model, checkpoint_path=None):
    """
    Evaluates the trained grid_model + classifier on the full MNIST test set.
    Loads from checkpoint_path if it exists.
    Does not do backprop; strictly forward pass to compute accuracy.
    """
    import torch
    # (A) Recreate the same classifier head to match training
    num_supernodes = grid_model.x_dim * grid_model.y_dim * grid_model.z_dim
    input_dim = num_supernodes * 9 * grid_model.out_channels
    classifier_head = nn.Linear(input_dim, 10)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Test phase: loading checkpoint '{checkpoint_path}'.")
        checkpoint = torch.load(checkpoint_path)
        grid_model.supernode_model.load_state_dict(checkpoint["supernode_model_state"])
        classifier_head.load_state_dict(checkpoint["classifier_head_state"])
    else:
        print("No checkpoint found for testing. Using current weights.")

    # Put model in eval mode so that any dropout or BN is disabled
    grid_model.supernode_model.eval()
    classifier_head.eval()

    # (B) Get the MNIST test set
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_test = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    from torch.utils.data import DataLoader
    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    total_correct = 0
    total_samples = 0

    # We do not need gradients for testing
    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Testing", leave=True):
            grid_model.reinitialize_grid()

            # Flatten each test image to [1 x 784], expand to [9 x 784]
            flattened_img = img.view(1, 28 * 28)
            expanded_img = flattened_img.expand(9, -1)
            # Set each supernode's x at t=0
            for z in range(grid_model.z_dim):
                for y in range(grid_model.y_dim):
                    for x in range(grid_model.x_dim):
                        grid_model.current_grid[(x, y, z, 0)].x = expanded_img.clone()

            # Forward pass
            grid_model.run_full_sequence()
            final_embeddings = grid_model.get_final_embeddings()
            flat_emb = final_embeddings.view(1, -1)
            logits = classifier_head(flat_emb)
            pred_label = logits.argmax(dim=1)

            total_correct += (pred_label == label).sum().item()
            total_samples += label.size(0)

    # Final accuracy
    test_acc = total_correct / total_samples
    print(f"Test Accuracy on entire MNIST test set: {test_acc:.4f}")

# If this file is run directly, execute main()
if __name__ == "__main__":
    main()
