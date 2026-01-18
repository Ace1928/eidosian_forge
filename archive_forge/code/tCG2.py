import unittest  # Importing the unittest module for creating and running tests
import torch  # Importing the PyTorch library for tensor computations and neural network operations
import sys  # Importing the sys module to interact with the Python runtime environment
import asyncio  # Importing the asyncio module for writing single-threaded concurrent code using coroutines
import logging  # Importing the logging module to enable logging of messages of various severity levels
from unittest.mock import (
    patch,
)  # Importing the patch function from unittest.mock to mock objects during tests
import numpy as np  # Importing the numpy library for numerical operations on arrays
import unittest
import numpy as np
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


# Append the system path to include the specific directory for module importation
sys.path.append(
    "/home/lloyd/EVIE/Indellama3/indego"
)  # Modifying sys.path to include the directory containing the Indego modules

# Importing specific classes and functions from the IndegoAdaptAct module
from ActivationDictionary import (
    ActivationDictionary,
)  # Importing the ActivationDictionary class which manages activation functions
from IndegoAdaptAct import (
    EnhancedPolicyNetwork,  # Importing the EnhancedPolicyNetwork class, a neural network for policy decisions
    AdaptiveActivationNetwork,  # Importing the AdaptiveActivationNetwork class, a neural network that adapts its activation functions
    calculate_reward,  # Importing the calculate_reward function to compute rewards in reinforcement learning scenarios
    update_policy_network,  # Importing the update_policy_network function to update the policy network based on rewards
    log_decision,  # Importing the log_decision function to log decisions made by the policy network
)

# Importing the configure_logging function from the IndegoLogging module to set up advanced logging configurations
from IndegoLogging import configure_logging


# Asynchronous setup of the logging module
async def setup_logging() -> None:
    """
    Asynchronously sets up logging configurations using the configure_logging function from the IndegoLogging module.
    This function is designed to be run within an asyncio event loop.
    """
    await configure_logging()  # Awaiting the completion of the configure_logging function which sets up logging based on a configuration file


# Ensuring that there is an event loop available for the current thread
try:
    loop = asyncio.get_event_loop()  # Attempting to get the current event loop
except (
    RuntimeError
) as e:  # Handling the RuntimeError that occurs if no event loop is present in the current thread
    if "There is no current event loop in thread" in str(
        e
    ):  # Checking if the error message indicates the absence of an event loop
        loop = asyncio.new_event_loop()  # Creating a new event loop
        asyncio.set_event_loop(
            loop
        )  # Setting the newly created event loop as the current event loop

# Utilizing asyncio's event loop to perform the asynchronous logging setup
if (
    asyncio.get_event_loop().is_running()
):  # Checking if the current event loop is already running
    asyncio.run(
        setup_logging()
    )  # If the event loop is running, perform the logging setup synchronously
else:
    loop = asyncio.get_event_loop()  # Getting the current event loop
    loop.run_until_complete(
        setup_logging()
    )  # Running the setup_logging function asynchronously until it completes

# Acquiring a logger instance for the current module from the centralized logging configuration
logger = logging.getLogger(
    __name__
)  # Getting a logger with the name of the current module, configured as per the IndegoLogging module's settings


class TestIndegoAdaptAct(unittest.TestCase):

    def test_enhanced_policy_network_initialization(self):
        input_dim = 5
        output_dim = 3
        layers_info = [10, 15]
        hyperparameters = {
            "learning_rate": 0.01,
            "regularization_factor": 0.001,
            "discount_factor": 0.99,
        }
        network = EnhancedPolicyNetwork(
            input_dim, output_dim, layers_info, hyperparameters
        )
        self.assertEqual(network.input_dim, input_dim)
        self.assertEqual(network.output_dim, output_dim)
        self.assertEqual(network.layers_info, layers_info)
        self.assertEqual(network.hyperparameters, hyperparameters)

    def test_enhanced_policy_network_forward_pass(self):
        input_dim = 5
        output_dim = 3
        layers_info = [10, 15]
        network = EnhancedPolicyNetwork(input_dim, output_dim, layers_info)
        input_tensor = torch.randn(16, input_dim)
        output_tensor = network(input_tensor)
        self.assertEqual(output_tensor.shape, (16, output_dim))

    def test_adaptive_activation_network_initialization(self):
        activation_dict = ActivationDictionary()
        in_features = 5
        out_features = 3
        layers_info = [10, 15]
        network = AdaptiveActivationNetwork(
            activation_dict, in_features, out_features, layers_info
        )
        self.assertEqual(network.activations, activation_dict.activation_types)
        self.assertEqual(
            network.activation_keys, list(activation_dict.activation_types.keys())
        )
        self.assertEqual(len(network.model), len(layers_info) + 1)
        self.assertEqual(
            network.policy_network.input_dim, in_features + sum(layers_info)
        )
        self.assertEqual(
            network.policy_network.output_dim, len(activation_dict.activation_types)
        )

    def test_adaptive_activation_network_forward_pass(self):
        activation_dict = ActivationDictionary()
        in_features = 5
        out_features = 3
        layers_info = [10, 15]
        network = AdaptiveActivationNetwork(
            activation_dict, in_features, out_features, layers_info
        )
        input_tensor = torch.randn(16, in_features)
        output_tensor = network(input_tensor)
        self.assertEqual(output_tensor.shape, (16, out_features))

    def test_calculate_reward(self):
        current_loss = 0.5
        previous_loss = 1.0
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        reward = calculate_reward(current_loss, previous_loss, y_true, y_pred)
        self.assertAlmostEqual(reward, 1.25)

    def test_update_policy_network(self):
        policy_network = EnhancedPolicyNetwork(5, 3, [10, 15])
        optimizer = optim.Adam(policy_network.parameters(), lr=0.01)
        reward = 1.0
        log_prob = torch.tensor(0.5, requires_grad=True)
        update_policy_network(policy_network, optimizer, reward, log_prob)

    def test_log_decision(self):
        layer_output = torch.tensor([0.1, 0.2, 0.3, 0.4])
        chosen_activation = "ReLU"
        reward = 1.0
        log_decision(layer_output, chosen_activation, reward)

    def test_enhanced_policy_network_initialization2(self):
        input_dim = 5
        output_dim = 3
        layers_info = [10, 15]
        hyperparameters = {
            "learning_rate": 0.01,
            "regularization_factor": 0.001,
            "discount_factor": 0.99,
        }
        network = EnhancedPolicyNetwork(
            input_dim, output_dim, layers_info, hyperparameters
        )
        self.assertEqual(network.input_dim, input_dim)
        self.assertEqual(network.output_dim, output_dim)
        self.assertEqual(network.layers_info, layers_info)
        self.assertEqual(network.hyperparameters, hyperparameters)

    def test_enhanced_policy_network_forward_pass2(self):
        input_dim = 5
        output_dim = 3
        layers_info = [10, 15]
        network = EnhancedPolicyNetwork(input_dim, output_dim, layers_info)
        input_tensor = torch.randn(16, input_dim)
        output_tensor = network(input_tensor)
        self.assertEqual(output_tensor.shape, (16, output_dim))

    def test_adaptive_activation_network_initialization2(self):
        activation_dict = ActivationDictionary()
        in_features = 5
        out_features = 3
        layers_info = [10, 15]
        network = AdaptiveActivationNetwork(
            activation_dict, in_features, out_features, layers_info
        )
        self.assertEqual(network.activations, activation_dict.activation_types)
        self.assertEqual(
            network.activation_keys, list(activation_dict.activation_types.keys())
        )
        self.assertEqual(len(network.model), len(layers_info) + 1)
        self.assertEqual(
            network.policy_network.input_dim, in_features + sum(layers_info)
        )
        self.assertEqual(
            network.policy_network.output_dim, len(activation_dict.activation_types)
        )

    def test_adaptive_activation_network_forward_pass2(self):
        activation_dict = ActivationDictionary()
        in_features = 5
        out_features = 3
        layers_info = [10, 15]
        network = AdaptiveActivationNetwork(
            activation_dict, in_features, out_features, layers_info
        )
        input_tensor = torch.randn(16, in_features)
        output_tensor = network(input_tensor)
        self.assertEqual(output_tensor.shape, (16, out_features))

    def test_calculate_reward2(self):
        current_loss = 0.5
        previous_loss = 1.0
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        reward = calculate_reward(current_loss, previous_loss, y_true, y_pred)
        self.assertAlmostEqual(reward, 1.25)

    def test_update_policy_network2(self):
        policy_network = EnhancedPolicyNetwork(5, 3, [10, 15])
        optimizer = optim.Adam(policy_network.parameters(), lr=0.01)
        reward = 1.0
        log_prob = torch.tensor(0.5, requires_grad=True)
        update_policy_network(policy_network, optimizer, reward, log_prob)

    def test_log_decision2(self):
        layer_output = torch.tensor([0.1, 0.2, 0.3, 0.4])
        chosen_activation = "ReLU"
        reward = 1.0
        log_decision(layer_output, chosen_activation, reward)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
