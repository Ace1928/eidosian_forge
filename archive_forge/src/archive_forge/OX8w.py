import logging
import torch
import pandas as pd
import concurrent.futures
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Dict, Any, Optional
import sys
import os
import importlib.util

# Constructing the path to the ActivationDictionary.py file
module_name = "ActivationDictionary"
file_path = os.path.join(os.path.dirname(__file__), f"{module_name}.py")

# Dynamically loading the module from the given file path
spec = importlib.util.spec_from_file_location(module_name, file_path)
if spec and spec.loader:
    activation_dictionary_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(activation_dictionary_module)
    # Retrieving the activation types dictionary from the loaded module
    self.activation_types = activation_dictionary_module.activation_types
    logging.info("Activation types loaded successfully.")
else:
    logging.error(
        "Failed to load the module for activation functions due to missing spec or loader."
    )
    raise ImportError("The activation function module could not be loaded.")


class IndegoActivation:
    """
    Manages the activation functions within neural networks, ensuring a comprehensive and robust selection tailored to various network layers. This class encapsulates the complexity of activation function dynamics and provides a systematic approach to their management and application, adhering to the highest standards of software engineering and mathematical precision.

    Attributes:
        activation_types (Dict[str, Callable[[float], float]]): A dictionary mapping activation function names to their mathematical representations, allowing for dynamic selection and application.
        current_activation (Optional[str]): The currently active activation function type, facilitating state tracking and operational consistency.

    Methods:
        initialize_activation_types(): Initializes the dictionary of activation functions with their respective mathematical implementations.
        apply_function(type: str, input: float) -> float: Applies the specified activation function to the input and returns the result, incorporating detailed logging and error handling.
    """

    def __init__(self):
        """
        Constructs an instance of the ActivationFunctionManager, meticulously setting up the foundational state and structure for managing a diverse array of activation functions utilized within neural networks. This constructor method is responsible for initializing the dictionary of activation functions, which encapsulates a variety of mathematical models tailored for neural computation. Additionally, it sets the initial state of the current activation to None, ensuring a clean slate for subsequent operations. This method adheres to the highest standards of software engineering, providing a robust and systematic approach to activation function management.

        The method performs the following operations:
        1. It calls the initialize_activation_types method to populate the activation_types dictionary with the respective lambda expressions representing the mathematical logic of each activation function.
        2. It initializes the current_activation attribute to None, establishing a neutral starting point for activation function application.
        3. It logs detailed information about the initialization process, specifically listing the supported activation function types, which enhances traceability and debugging capabilities.
        """
        self.initialize_activation_types()  # Populate the activation_types dictionary with mathematical implementations

        self.current_activation = None  # Initialize the current activation to None indicating no active function

        logging.info(
            "ActivationFunctionManager initialized with supported types: "
            + ", ".join(self.activation_types.keys())
        )

        def parallelized_computation(activation_dict):
            """
            Executes the initialization of activation functions in a parallelized manner using a ThreadPoolExecutor.
            This method ensures that each activation function is initialized concurrently, leveraging multi-threading
            to enhance performance and efficiency.

            Parameters:
                activation_dict (Dict[str, Callable[[float], float]]): A dictionary containing the activation function names and their corresponding
                                        lambda expressions for initialization.

            The method logs detailed information about each activation function as it is initialized, and captures
            and logs any exceptions that occur during the parallel execution process.
            """
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submitting tasks to the executor and mapping futures to activation function names
                future_to_activation = {
                    executor.submit(lambda x: activation_dict[x](), name): name
                    for name in activation_dict
                }
                for future in concurrent.futures.as_completed(future_to_activation):
                    activation_name = future_to_activation[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        logging.error(
                            f"{activation_name} generated an exception: {exc}"
                        )
                    else:
                        logging.info(f"{activation_name}: Computation result - {data}")

        # Logging the initialization of each activation function with detailed mathematical descriptions and expected input-output ranges.
        for func_name, func in self.activation_types.items():
            logging.debug(
                f"Initialized {func_name} activation function with lambda expression: {func}"
            )

        # Utilizing parallel computation to initialize and log activation functions
        parallelized_computation(self.activation_types)

    def apply_function(self, type: str, input: float) -> float:
        """
        Applies the specified activation function to the given input using advanced mathematical models. This method includes comprehensive error handling to ensure that only supported activation types are used, and it logs detailed information about the application process.

        Parameters:
            type (str): The type of activation function to apply. Must be one of the supported types defined in the activation_types dictionary.
            input (float): The input value to which the activation function will be applied.

        Returns:
            float: The output from the activation function, calculated using the appropriate mathematical model.

        Raises:
            ValueError: If the specified activation type is not supported, an error is logged and a ValueError is raised to prevent misuse of the function.
        """
        if type not in self.activation_types:
            error_message = f"Unsupported activation type '{type}'. Available types: {', '.join(self.activation_types.keys())}."
            logging.error(error_message)
            raise ValueError(error_message)

        logging.info(f"Applying {type} activation function to input: {input}")

        result = self.activation_types[type](input)

        self.current_activation = type

        logging.debug(
            f"{type} activation function applied to input {input}, resulting in output {result}"
        )

        return result
