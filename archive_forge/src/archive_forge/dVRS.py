import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import zipfile
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# Download and prepare the dataset
dataset_url = "https://raw.githubusercontent.com/PolyAI-LDN/conversational-datasets/master/insuranceQA/train.json"
dataset_path = "/home/lloyd/Downloads/indegodata/datasets/insuranceQA_train.json"

if not os.path.exists(dataset_path):
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    response = requests.get(dataset_url)
    with open(dataset_path, "wb") as f:
        f.write(response.content)

# Load the dataset
data = pd.read_json(dataset_path)
questions = data["question"].tolist()
answers = data["answer"].tolist()

# Tokenize and preprocess the dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class TextDataset(Dataset):
    """
    Custom dataset class for text-based chat conversation.
    """

    def __init__(self, questions, answers, tokenizer, max_length=128):
        """
        Initialize the TextDataset.

        Parameters:
        questions (list): List of questions.
        answers (list): List of answers.
        tokenizer (BertTokenizer): Tokenizer for text preprocessing.
        max_length (int): Maximum length of tokenized sequences.
        """
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
        int: Number of samples.
        """
        return len(self.questions)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters:
        idx (int): Index of the sample.

        Returns:
        dict: Tokenized input and label tensors.
        """
        question = self.questions[idx]
        answer = self.answers[idx]
        inputs = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }


# Split the dataset into training and validation sets
train_questions, val_questions, train_answers, val_answers = train_test_split(
    questions, answers, test_size=0.2, random_state=42
)

# Create DataLoader for training and validation sets
train_dataset = TextDataset(train_questions, train_answers, tokenizer)
val_dataset = TextDataset(val_questions, val_answers, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define Dynamic Activation Neurons (DANs)
class DAN(nn.Module):
    """
    Dynamic Activation Neurons (DANs) class.
    This class defines a neural network module that applies a linear transformation followed by a ReLU activation function.
    The output is scaled by a learnable parameter.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the DAN module.

        Parameters:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        """
        super(DAN, self).__init__()
        self.basis_function = nn.Linear(
            input_size, output_size
        )  # Linear transformation layer
        self.param = nn.Parameter(
            torch.randn(output_size)
        )  # Learnable parameter for scaling the output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DAN module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the linear transformation, ReLU activation, and scaling.
        """
        return self.param * F.relu(
            self.basis_function(x)
        )  # Apply linear transformation, ReLU, and scale


# Define Dynamic Activation Synapses (DASs)
class DAS(nn.Module):
    """
    Dynamic Activation Synapses (DASs) class.
    This class defines a neural network module that applies a linear transformation followed by a sigmoid activation function.
    The output is scaled by a learnable parameter.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the DAS module.

        Parameters:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        """
        super(DAS, self).__init__()
        self.basis_function = nn.Linear(
            input_size, output_size
        )  # Linear transformation layer
        self.param = nn.Parameter(
            torch.randn(output_size)
        )  # Learnable parameter for scaling the output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DAS module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the linear transformation, sigmoid activation, and scaling.
        """
        return self.param * torch.sigmoid(
            self.basis_function(x)
        )  # Apply linear transformation, sigmoid, and scale


# Define Hexagonal Topology Network (HTN)
class HTN(nn.Module):
    """
    Hexagonal Topology Network (HTN) class.
    This class defines a neural network with a hexagonal topology using alternating DAN and DAS modules.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the HTN module.

        Parameters:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        """
        super(HTN, self).__init__()
        # Initialize six DAN and DAS modules in alternating order
        self.dan1 = DAN(input_size, output_size)
        self.das1 = DAS(output_size, output_size)
        self.dan2 = DAN(output_size, output_size)
        self.das2 = DAS(output_size, output_size)
        self.dan3 = DAN(output_size, output_size)
        self.das3 = DAS(output_size, output_size)
        self.dan4 = DAN(output_size, output_size)
        self.das4 = DAS(output_size, output_size)
        self.dan5 = DAN(output_size, output_size)
        self.das5 = DAS(output_size, output_size)
        self.dan6 = DAN(output_size, output_size)
        self.das6 = DAS(output_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the HTN module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the hexagonal topology network.
        """
        # Sequentially pass the input through the DAN and DAS modules
        o1 = self.dan1(x)
        o2 = self.das1(o1)
        o3 = self.dan2(o2)
        o4 = self.das2(o3)
        o5 = self.dan3(o4)
        o6 = self.das3(o5)
        o7 = self.dan4(o6)
        o8 = self.das4(o7)
        o9 = self.dan5(o8)
        o10 = self.das5(o9)
        o11 = self.dan6(o10)
        return o11  # Final output


# Define Dynamic Fractal Topology Output (DFTO)
class DFTO(nn.Module):
    """
    Dynamic Fractal Topology Output (DFTO) class.
    This class defines a neural network module that applies various aggregation operations on the input tensor.
    """

    def __init__(self, operation: str = "sum"):
        """
        Initialize the DFTO module.

        Parameters:
        operation (str): The aggregation operation to apply. Options are 'sum', 'max', 'avg', 'min', 'aggregate', 'power_spectrum'.
        """
        super(DFTO, self).__init__()
        self.operation = operation  # Store the specified operation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DFTO module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the specified aggregation operation.
        """
        if self.operation == "sum":
            return torch.sum(x, dim=1)  # Sum along the specified dimension
        elif self.operation == "max":
            return torch.max(x, dim=1)[0]  # Max along the specified dimension
        elif self.operation == "avg":
            return torch.mean(x, dim=1)  # Mean along the specified dimension
        elif self.operation == "min":
            return torch.min(x, dim=1)[0]  # Min along the specified dimension
        elif self.operation == "aggregate":
            return torch.sum(x, dim=1) / x.size(
                1
            )  # Aggregate by summing and dividing by the size
        elif self.operation == "power_spectrum":
            return torch.sum(
                x**2, dim=1
            )  # Power spectrum by summing the squares along the specified dimension


# Integrate Components into a Full Model
class FullHTNModel(nn.Module):
    """
    Full Hexagonal Topology Network Model (FullHTNModel) class.
    This class integrates the HTN and DFTO modules into a complete model.
    """

    def __init__(self, input_size: int, output_size: int, operation: str = "sum"):
        """
        Initialize the FullHTNModel module.

        Parameters:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        operation (str): The aggregation operation to apply in the DFTO module.
        """
        super(FullHTNModel, self).__init__()
        self.htn = HTN(input_size, output_size)  # Initialize the HTN module
        self.dfto = DFTO(operation)  # Initialize the DFTO module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FullHTNModel module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the HTN and DFTO modules.
        """
        htn_output = self.htn(x)  # Pass input through HTN
        dfto_output = self.dfto(htn_output)  # Pass HTN output through DFTO
        return dfto_output  # Final output


# Training and Optimization Function
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
):
    """
    Train the model using the provided data loader, criterion, and optimizer.

    Parameters:
    model (nn.Module): The neural network model to train.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
    val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
    criterion (nn.Module): Loss function.
    optimizer (torch.optim.Optimizer): Optimization algorithm.
    epochs (int): Number of training epochs.
    """
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0.0  # Initialize total loss for the epoch
        for batch in train_loader:
            inputs = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters
            total_loss += loss.item()  # Accumulate loss
        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss / len(train_loader)}"
        )  # Print epoch loss

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(
            f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss / len(val_loader)}"
        )  # Print validation loss
        model.train()  # Set the model back to training mode


# Example Usage
if __name__ == "__main__":
    # Initialize model, criterion, and optimizer
    input_size = 128
    output_size = 64
    model = FullHTNModel(input_size, output_size)  # Initialize the full model
    criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # Train the model
    train_model(
        model, train_loader, val_loader, criterion, optimizer, epochs=10
    )  # Train the model for 10 epochs
