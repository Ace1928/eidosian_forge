import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Callable, Dict, Optional, Union, Any
import logging

        Retrieves an activation function by name, ensuring a robust and detailed logging of the retrieval process.

        Parameters:
            name (str): The name of the activation function to retrieve, which must be a string representing the key in the activation_types dictionary.

        Returns:
            Optional[DynamicActivation]: The activation function as a lambda expression if found, otherwise None. The return type is meticulously annotated to ensure clarity in expected output.

        Raises:
            KeyError: If the name provided does not correspond to any existing activation function, a KeyError is raised with a detailed error message.
        