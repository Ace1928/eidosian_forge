import math
import logging


class ActivationFunctionManager:
    """
    Manages the activation functions within neural networks, ensuring a comprehensive and robust selection tailored to various network layers. This class encapsulates the complexity of activation function dynamics and provides a systematic approach to their management and application, adhering to the highest standards of software engineering and mathematical precision.

    Attributes:
        activation_types (dict): A dictionary mapping activation function names to their mathematical representations, allowing for dynamic selection and application.
        current_activation (str): The currently active activation function type, facilitating state tracking and operational consistency.

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
        # Invoke the method to initialize the dictionary of activation functions with their respective mathematical implementations.
        self.initialize_activation_types()

        # Set the initial state of the current activation function type to None to indicate that no activation function is currently active.
        self.current_activation = None

        # Log the initialization details with high verbosity, listing all supported activation function types to provide a clear overview of the available options.
        logging.info(
            "ActivationFunctionManager initialized with supported types: "
            + ", ".join(self.activation_types.keys())
        )

    def initialize_activation_types(self):
        """
        Sets up the dictionary of activation functions with their respective lambda expressions, encapsulating the mathematical logic required for each function. This method enhances modularity and maintainability of the activation function management.
        """
        self.activation_types = {
            "ReLU": lambda x: max(0, x),
            "Sigmoid": lambda x: 1 / (1 + math.exp(-x)),
            "Tanh": lambda x: math.tanh(x),
        }

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
            error_message = f"Attempted to use unsupported activation type: {type}"
            logging.error(error_message)
            raise ValueError(error_message)

        self.current_activation = type
        result = self.activation_types[type](input)
        logging.debug(
            f"Applied {type} activation function to input {input}, resulting in output {result}"
        )
        return result
