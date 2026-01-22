import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import numpy as np
import asyncio
from typing import Dict, Any, List, cast, Tuple, Callable, Optional, Union
import sys
import torch.optim as optim
import os
from ActivationDictionary import ActivationDictionary
from IndegoLogging import configure_logging

        Dynamically selects and applies activation functions based on the policy network's output,
        with an option for deterministic or probabilistic selection of activations.

        Args:
            x (torch.Tensor): The input tensor to be processed through activation functions.
            batch_id (Optional[int]): An identifier for the batch, used for logging purposes.
            deterministic (bool): If True, the activation function with the highest score is selected,
                                  otherwise, a probabilistic approach is used.

        Returns:
            torch.Tensor: The tensor resulting from applying the selected activation functions,
                          with padding applied to ensure all output tensors have the same size.
        