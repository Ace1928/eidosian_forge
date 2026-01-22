import unittest
import torch
import sys
import asyncio
import logging
from unittest.mock import patch
import numpy as np

# Append the system path to include the specific directory for module importation
sys.path.append("/home/lloyd/EVIE/Indellama3/indego")
from ActivationDictionary import ActivationDictionary

# Import the IndegoAdaptAct module
from IndegoAdaptAct import (
    EnhancedPolicyNetwork,
    AdaptiveActivationNetwork,
    calculate_reward,
    update_policy_network,
    log_decision,
)

# Integrating the advanced logging configuration from the IndegoLogging module to ensure all events are meticulously recorded with utmost detail and precision
from IndegoLogging import configure_logging


# Asynchronous setup of the logging module
async def setup_logging():
    await configure_logging()  # This function call configures the logging based on the IndegoLogging module's configuration file


# Ensure there is an event loop for the current thread
try:
    loop = asyncio.get_event_loop()
except RuntimeError as e:
    if "There is no current event loop in thread" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
# Utilizing asyncio's event loop to perform the asynchronous logging setup
# Check that there is a main loop, if there is not (ie during testing) ensure that one is started and is closed once logging finishes.
if asyncio.get_event_loop().is_running():
    # If the event loop is already running, run the logging setup synchronously
    asyncio.run(setup_logging())
else:
    # If the event loop is not running, start it and run the logging setup asynchronously
    loop = asyncio.get_event_loop()
    loop.run_until_complete(setup_logging())
# Acquiring a logger instance for the current module from the centralized logging configuration
# This logger adheres to the configurations set up by the IndegoLogging module, ensuring all logging is centralized, easily manageable, and aligned with the highest standards of operational excellence
logger = logging.getLogger(__name__)


class TestIndegoAdaptAct(unittest.TestCase):

    def test_enhanced_policy_network(self):
        policy_network = EnhancedPolicyNetwork(
            input_dim=10, output_dim=5, layers_info=[32, 32]
        )
        input_tensor = torch.randn(16, 10)
        output = policy_network(input_tensor)
        self.assertEqual(output.shape, torch.Size([16, 5]))

    def test_adaptive_activation_network(self):
        activation_dict = ActivationDictionary()
        adaptive_network = AdaptiveActivationNetwork(
            activation_dict, in_features=10, out_features=5, layers_info=[32, 32]
        )
        input_tensor = torch.randn(16, 10)
        output = adaptive_network(input_tensor)
        self.assertEqual(output.shape, torch.Size([16, 5]))

    def test_calculate_reward(self):
        current_loss = 0.5
        previous_loss = 0.6
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        reward = calculate_reward(current_loss, previous_loss, y_true, y_pred)
        self.assertAlmostEqual(reward, 1.1)

    def test_update_policy_network(self):
        policy_network = EnhancedPolicyNetwork(
            input_dim=10, output_dim=5, layers_info=[32, 32]
        )
        optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.01)
        reward = 1.0
        log_prob = 0.5
        update_policy_network(policy_network, optimizer, reward, log_prob)

    def test_log_decision(self):
        layer_output = torch.randn(16, 32)
        chosen_activation = "ReLU"
        reward = 1.0
        log_decision(layer_output, chosen_activation, reward)


if __name__ == "__main__":
    unittest.main()
