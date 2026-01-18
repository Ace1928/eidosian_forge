import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
import logging

# Define the local paths for the dataset
train_data_path = "/media/lloyd/Aurora_M2/indegodata/processed_data/universal_train.csv"
val_data_path = (
    "/media/lloyd/Aurora_M2/indegodata/processed_data/universal_validation.csv"
)
test_data_path = "/media/lloyd/Aurora_M2/indegodata/processed_data/universal_test.csv"

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to load datasets
def load_dataset(path, dtype_dict=None, low_memory=True):
    if dtype_dict:
        return pd.read_csv(path, dtype=dtype_dict, low_memory=low_memory)
    else:
        return pd.read_csv(path, low_memory=low_memory)


dtype_dict = {"question": str, "answer": str}

# Load the datasets into pandas DataFrame
train_data = load_dataset(train_data_path, dtype_dict, low_memory=False)
val_data = load_dataset(val_data_path, dtype_dict, low_memory=False)
test_data = load_dataset(test_data_path, dtype_dict, low_memory=False)


# Function to extract questions and answers
def extract_qa(data):
    try:
        questions = data["question"].tolist()
        answers = data["answer"].tolist()
        logger.info("Questions and answers extracted successfully")
        return questions, answers
    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        raise


# Extract questions and answers from the datasets
train_questions, train_answers = extract_qa(train_data)
val_questions, val_answers = extract_qa(val_data)
test_questions, test_answers = extract_qa(test_data)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def validate_data(data: pd.DataFrame, required_columns: list) -> None:
    """
    Validate that the required columns exist in the DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame to validate.
    required_columns (list): List of required column names.
    """
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Missing required column: {column}")


required_columns = ["question", "answer"]
validate_data(train_data, required_columns)
validate_data(val_data, required_columns)
validate_data(test_data, required_columns)


def safe_tokenize(tokenizer, text, max_length):
    """
    Safely tokenize the input text.

    Parameters:
    tokenizer (BertTokenizer): Tokenizer for text preprocessing.
    text (str): Input text to tokenize.
    max_length (int): Maximum length of tokenized sequences.

    Returns:
    dict: Tokenized input.
    """
    if not isinstance(text, str):
        text = str(text)  # Convert non-str data to string
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


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
        self.questions = [str(q) for q in questions]
        self.answers = [str(a) for a in answers]
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
        inputs = safe_tokenize(self.tokenizer, question, self.max_length)
        labels = safe_tokenize(self.tokenizer, answer, self.max_length)
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }


# Create DataLoader for training and validation sets
train_dataset = TextDataset(train_questions, train_answers, tokenizer)
val_dataset = TextDataset(val_questions, val_answers, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


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
        self.basis_function = nn.Linear(input_size, output_size)
        self.param = nn.Parameter(torch.randn(output_size, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DAN module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the linear transformation, ReLU activation, and scaling.
        """
        x = x.float()  # Ensure input is float
        return self.param * F.relu(self.basis_function(x))


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
        self.basis_function = nn.Linear(input_size, output_size)
        self.param = nn.Parameter(torch.randn(output_size, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DAS module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the linear transformation, sigmoid activation, and scaling.
        """
        x = x.float()  # Ensure input is float
        return self.param * torch.sigmoid(self.basis_function(x))


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
        x = x.float()  # Ensure input is float
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
        return o11


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
        self.operation = operation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DFTO module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the specified aggregation operation.
        """
        x = x.float()  # Ensure input is float
        if self.operation == "sum":
            return torch.sum(x, dim=1)
        elif self.operation == "max":
            return torch.max(x, dim=1)[0]
        elif self.operation == "avg":
            return torch.mean(x, dim=1)
        elif self.operation == "min":
            return torch.min(x, dim=1)[0]
        elif self.operation == "aggregate":
            return torch.sum(x, dim=1) / x.size(1)
        elif self.operation == "power_spectrum":
            return torch.sum(x**2, dim=1)


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
        self.htn = HTN(input_size, output_size)
        self.dfto = DFTO(operation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FullHTNModel module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the HTN and DFTO modules.
        """
        htn_output = self.htn(x)
        dfto_output = self.dfto(htn_output)
        return dfto_output


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    save_path: str = "model.pth",
) -> None:
    """
    Train the model using the provided data loader, criterion, and optimizer.

    Parameters:
    model (nn.Module): The neural network model to train.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
    val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
    criterion (nn.Module): Loss function.
    optimizer (torch.optim.Optimizer): Optimization algorithm.
    epochs (int): Number of training epochs.
    save_path (str): Path to save the trained model.
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            try:
                inputs = batch["input_ids"].float()
                labels = batch["labels"].float()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                logger.error(f"Training error at epoch {epoch}: {e}")
                continue
        logger.info(f"Epoch {epoch+1}: Training Loss: {total_loss / len(train_loader)}")
        validate_model(model, val_loader, criterion)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")


def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            try:
                inputs = batch["input_ids"].float()
                labels = batch["labels"].float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            except Exception as e:
                logger.error(f"Validation error: {e}")
                continue
    logger.info(f"Validation Loss: {val_loss / len(val_loader)}")


def save_model(model: nn.Module, path: str):
    """
    Save the model to the specified path.

    Parameters:
    model (nn.Module): The neural network model to save.
    path (str): Path to save the model.
    """
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def load_model(model: nn.Module, path: str):
    """
    Load the model from the specified path.

    Parameters:
    model (nn.Module): The neural network model to load.
    path (str): Path to load the model from.
    """
    model.load_state_dict(torch.load(path))
    model.eval()
    logger.info(f"Model loaded from {path}")


def predict(
    model: nn.Module, tokenizer: BertTokenizer, text: str, max_length: int = 128
) -> torch.Tensor:
    """
    Predict the output for the given text using the trained model.

    Parameters:
    model (nn.Module): The trained neural network model.
    tokenizer (BertTokenizer): Tokenizer for text preprocessing.
    text (str): Input text for prediction.
    max_length (int): Maximum length of tokenized sequences.

    Returns:
    torch.Tensor: Predicted output tensor.
    """
    model.eval()
    inputs = safe_tokenize(tokenizer, text, max_length)
    with torch.no_grad():
        outputs = model(inputs["input_ids"].float())  # Ensure inputs are float
    return outputs


if __name__ == "__main__":
    input_size = 128
    output_size = 64
    model = FullHTNModel(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    save_model(model, "untrained_model.pth")

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=10,
        save_path="trained_model.pth",
    )

    load_model(model, "trained_model.pth")

    sample_text = "What is the process to claim insurance?"
    prediction = predict(model, tokenizer, sample_text)
    logger.info(f"Prediction: {prediction}")
