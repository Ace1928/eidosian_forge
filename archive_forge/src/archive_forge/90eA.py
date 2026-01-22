import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
from collections import deque
from matplotlib.widgets import Button, CheckButtons, Slider
from matplotlib.animation import FuncAnimation
import logging
import datetime
import sys
import cProfile
import pstats

# Configure logging
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file_path = "C:/Users/ace19/Documents/log_{}.txt".format(timestamp)
# Configure logging at the start of your script
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def log_exception(exc_type, exc_value, exc_traceback):
    """
    Logs uncaught exceptions with detailed information including type, value, and traceback.

    Parameters:
    - exc_type: Exception type.
    - exc_value: Exception value.
    - exc_traceback: Exception traceback object.
    """
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Set the logging for uncaught exceptions
sys.excepthook = log_exception

class SimulationParameters:
    """
    A class to hold and manage simulation parameters for the neural network.

    Attributes:
    - Various parameters controlling the simulation behavior and neural network characteristics.
    """

    def __init__(self):
        """
        Initializes the simulation parameters with default values.
        """
        self.signal_processing_mode = 'complex'
        self.signal_amplification = 1.0
        self.pattern_detection_sensitivity = 1.0
        self.filter_params = {'filter_type': 'low_pass', 'cutoff_frequency': 100}
        self.wavelet_params = {'wavelet_type': 'db1', 'level': 2}
        self.pattern_params = {'max_score': 50}
        self.mapping_params = {'steepness': 1.0, 'skew_factor': 1.0}
        self.output_params = {'expansion_factor': 1.0}
        self.input_layer_neuron_count = 7  # Example value, set as per your network design
        self.neuron_signal_size = 6  # Example value, representing the size of the signal for each neuron

    def update_params(self, **kwargs):
        """
        Dynamically updates simulation parameters with new values provided as keyword arguments.

        Parameters:
        - kwargs: A dictionary of parameter names and their new values.
        """
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            
class DynamicRingBuffer:
    """
    Implements a dynamic ring buffer using a deque, with automatic removal of oldest items.
    """

    def __init__(self, initial_size=1000):
        """
        Initializes the ring buffer with a specified initial size.

        Parameters:
        - initial_size: The maximum length of the deque.
        """
        self.data = deque(maxlen=initial_size)

    def append(self, item):
        """
        Appends an item to the deque, automatically removing the oldest item if the maximum length is exceeded.

        Parameters:
        - item: The item to be appended to the deque.
        """
        self.data.append(item)

    def get(self):
        """
        Returns the contents of the deque as a list.

        Returns:
        A list containing all items in the deque.
        """
        return list(self.data)
            
class SimplifiedNeuronWithDelay:
    """
    Represents a simplified model of a neuron with delay, including various parameters and methods for signal processing.
    """

    @staticmethod
    def generate_random_param(mean, sigma):
        """
        Generates a random parameter within a specific range using a normal distribution.

        Parameters:
        - mean: The mean of the normal distribution.
        - sigma: The standard deviation of the normal distribution.

        Returns:
        A clipped random value within the range (-128, 127).
        """
        return np.clip(np.random.normal(mean, sigma), -128, 127)

    def __init__(self, threshold, over_threshold, base_signal_strength, refractory_period, spike_threshold, scaling_factor, neuron_state, damping_factor, num_sub_windows, amplification_factor, temporal_window_size, pattern_params, mapping_params, output_params, simulation_params):
        """
        Initializes the neuron with various parameters and validates them.

        Parameters:
        - threshold: Activation threshold of the neuron.
        - over_threshold: Threshold for over-activation.
        - base_signal_strength: Base strength of the neuron's signal.
        - refractory_period: Period after activation during which the neuron cannot activate again.
        - spike_threshold: Threshold for generating a spike.
        - scaling_factor: Factor for scaling the input signal.
        - neuron_state: The state of the neuron (excitatory, inhibitory, neutral).
        - damping_factor: Factor for damping the signal.
        - num_sub_windows: Number of sub-windows for processing.
        - amplification_factor: Factor for amplifying the signal.
        - temporal_window_size: Size of the temporal window for signal processing.
        - pattern_params: Parameters for pattern detection.
        - mapping_params: Parameters for mapping neuron states.
        - output_params: Parameters for output signal processing.
        - simulation_params: Global simulation parameters.
        """
        self.threshold = self._validate_param(threshold, 'Threshold', int, float, min_val=0, max_val=127)
        self.over_threshold = self._validate_param(over_threshold, 'Over Threshold', int, float, min_val=0, max_val=127)
        self.base_signal_strength = self._validate_param(base_signal_strength, 'Base Signal Strength', int, float, min_val=1, max_val=128)
        self.refractory_period = self._validate_param(refractory_period, 'Refractory Period', int, float, min_val=1, max_val=5)
        self.spike_threshold = self._validate_param(spike_threshold, 'Spike Threshold', int, float, min_val=1, max_val=127)
        self.scaling_factor = self._validate_param(scaling_factor, 'Scaling Factor', int, float)
        self.neuron_state = self._validate_choice(neuron_state, 'Neuron State', ['excitatory', 'inhibitory', 'neutral'])
        self.damping_factor = self._validate_param(damping_factor, 'Damping Factor', int, float, min_val=1)
        self.num_sub_windows = self._validate_param(num_sub_windows, 'Number of Sub-windows', int, min_val=1)
        self.amplification_factor = self._validate_param(amplification_factor, 'Amplification Factor', int, float, min_val=1)
        self.temporal_window_size = self._validate_param(temporal_window_size, 'Temporal Window Size', int, min_val=1)

        # Initialize deque for input history with a fixed size
        self.input_history = deque(maxlen=temporal_window_size)

        # Additional parameters related to neuron behavior
        self.last_spike_time = -1
        self.signal_processing_mode = simulation_params.signal_processing_mode
        self.pattern_params = pattern_params
        self.mapping_params = mapping_params
        self.output_params = output_params

        # Generate random parameters
        self.spike_magnitude = self.generate_random_param(30, 5)
        self.pattern_detection_threshold = self.generate_random_param(0.5, 0.1)
        self.mapping_steepness = self.generate_random_param(0.1, 0.02)
        self.mapping_skew_factor = self.generate_random_param(0.1, 0.02)

        # Validate the types of pattern_params, mapping_params, and output_params
        self.pattern_params = self._validate_param(pattern_params, 'Pattern Params', dict)
        self.mapping_params = self._validate_param(mapping_params, 'Mapping Params', dict)
        self.output_params = self._validate_param(output_params, 'Output Params', dict)

    def _validate_param(self, param, name, *types, min_val=None, max_val=None):
        """
        Validates a parameter with type checking and range validation if applicable.

        Parameters:
        - param: The parameter to validate.
        - name: The name of the parameter (for error messages).
        - types: Allowed types for the parameter.
        - min_val: Optional minimum value for the parameter.
        - max_val: Optional maximum value for the parameter.

        Returns:
        The validated parameter.
        """
        if not isinstance(param, types):
            raise TypeError(f"{name} must be of type {types}")
        if min_val is not None and param < min_val:
            raise ValueError(f"{name} must be at least {min_val}")
        if max_val is not None and param > max_val:
            raise ValueError(f"{name} must not exceed {max_val}")
        return param

    def _validate_choice(self, choice, name, valid_choices):
        """
        Validates if the choice is among the valid options.

        Parameters:
        - choice: The choice to validate.
        - name: The name of the choice parameter (for error messages).
        - valid_choices: A list of valid options for the choice.

        Returns:
        The validated choice.
        """
        if choice not in valid_choices:
            raise ValueError(f"{name} must be one of {valid_choices}")
        return choice
       
    def advanced_signal_processing(self, signals, simulation_params):
        """
        Advanced signal processing logic based on simulation parameters.

        Parameters:
        - signals: The input signals to process.
        - simulation_params: The simulation parameters affecting processing.

        Returns:
        Processed signals after applying advanced signal processing logic.
        """
        self._validate_signals(signals)
        return signals * 2

    def basic_signal_processing(self, signals, simulation_params):
        """
        Basic signal processing, such as a simple linear transformation.

        Parameters:
        - signals: The input signals to process.
        - simulation_params: The simulation parameters affecting processing.

        Returns:
        Processed signals after applying basic signal processing logic.
        """
        self._validate_signals(signals)
        return signals * simulation_params.signal_amplification

    def spike_response(self):
        """
        Generate a response when the neuron spikes.

        Returns:
        The magnitude of the spike response.
        """
        return self.spike_magnitude

    def detect_pattern(self, input_history, processing_params):
        """
        Detects patterns in the input history based on processing parameters.

        Parameters:
        - input_history: The history of input signals.
        - processing_params: Parameters affecting pattern detection.

        Returns:
        A boolean indicating whether a pattern was detected.
        """
        if not isinstance(processing_params, dict):
            raise TypeError("Processing parameters must be a dictionary")
        pattern_strength = self.calculate_pattern_strength(input_history, processing_params)
        return pattern_strength > self.pattern_detection_threshold * processing_params["pattern_detection_sensitivity"]

    def calculate_pattern_strength(self, input_history, processing_params):
        """
        Calculates the strength of detected patterns in the input history.

        Parameters:
        - input_history: The history of input signals.
        - processing_params: Parameters affecting pattern detection.

        Returns:
        An integer representing the strength of detected patterns.
        """
        if len(input_history) == 0:
            return 0
        input_history_array = np.array(input_history)
        fft_result = np.fft.fft(input_history_array, axis=0)
        dominant_frequencies = np.abs(fft_result).mean(axis=0)
        entropy = -np.sum(dominant_frequencies * np.log(dominant_frequencies + 1e-9), axis=0)
        combined_score = np.sum(dominant_frequencies) + entropy
        normalized_score = np.clip(combined_score / processing_params["pattern_params"]["max_score"], 0, 1)

        # If normalized_score is an array, take the mean (or another aggregation method)
        # and then convert to an integer
        if isinstance(normalized_score, np.ndarray):
            normalized_score = int(np.mean(normalized_score))
        else:
            normalized_score = int(normalized_score)

        return normalized_score

    def state_mapping_function(self, input_signal_sum, mapping_params):
        """
        Maps the input signal sum to a neuron state using a sigmoid function and additional mapping parameters.

        Parameters:
        - input_signal_sum: The sum of input signals to the neuron.
        - mapping_params: Parameters for mapping the input signal to neuron states.

        Returns:
        An integer representing the mapped neuron state.
        """
        default_steepness = 1.0
        default_skew_factor = 1.0

        steepness = self.mapping_steepness * mapping_params.get("steepness", default_steepness)
        skew_factor = self.mapping_skew_factor * mapping_params.get("skew_factor", default_skew_factor)

        sigmoid = 1 / (1 + np.exp(-steepness * (input_signal_sum - self.threshold)))
        skewed_value = np.log1p(abs(input_signal_sum)) ** skew_factor
        combined_value = sigmoid * skewed_value
        mapped_state = np.clip(int(combined_value * 6), -6, 6)

        return mapped_state

    def granular_output(self, state, output_params):
        """
        Generates a granular output based on the neuron's state and output parameters.

        Parameters:
        - state: The current state of the neuron.
        - output_params: Parameters for generating the output.

        Returns:
        The granular output of the neuron.
        """
        expanded_output = state * output_params["expansion_factor"]
        return expanded_output

    def process_input(self, input_signals, current_time, simulation_params):
        """
        Processes input signals based on the current time and simulation parameters.

        Parameters:
        - input_signals: The input signals to process.
        - current_time: The current time step in the simulation.
        - simulation_params: The simulation parameters affecting processing.

        Returns:
        The output of the neuron after processing the input signals.
        """
        self._validate_process_input_params(input_signals, current_time, simulation_params)

        processed_signals = self._select_signal_processing_mode(input_signals, simulation_params)

        if current_time - self.last_spike_time < self.refractory_period:
            return 0
        total_signal = np.sum(processed_signals) * simulation_params.signal_amplification
        if total_signal > self.spike_threshold:
            self.last_spike_time = current_time
            return self.spike_response()

        scaled_signal = total_signal * self.scaling_factor
        damped_signal = scaled_signal // self.damping_factor
        self.input_history.append(input_signals)
        processing_params = {
            "pattern_detection_sensitivity": simulation_params.pattern_detection_sensitivity,
            "pattern_params": self.pattern_params
        }
        if self.detect_pattern(self.input_history, processing_params):
            damped_signal *= 1.2
        mapped_state = self.state_mapping_function(damped_signal, self.mapping_params)
        output = mapped_state * np.sign(total_signal) * self.amplification_factor
        return self.granular_output(output, self.output_params)

    # Helper methods for validation
    def _validate_signals(self, signals):
        """
        Validates that 'signals' is a NumPy array with appropriate dimensions.

        Parameters:
        - signals: The signals to validate.

        Raises:
        TypeError: If 'signals' is not a NumPy array.
        ValueError: If 'signals' array is not 2-dimensional.
        """
        if not isinstance(signals, np.ndarray):
            raise TypeError("Signals must be a NumPy array")
        if signals.ndim != 2:
            raise ValueError("Signals array must be 2-dimensional")

    def _validate_process_input_params(self, input_signals, current_time, simulation_params):
        """
        Validates input parameters for the process_input method.

        Parameters:
        - input_signals: The input signals to validate.
        - current_time: The current time step to validate.
        - simulation_params: The simulation parameters to validate.

        Raises:
        TypeError: If any parameter is of an incorrect type.
        """
        self._validate_signals(input_signals)
        if not isinstance(current_time, (int, float)):
            raise TypeError("Current time must be a number")
        if not isinstance(simulation_params, SimulationParameters):
            raise TypeError("Simulation parameters must be an instance of SimulationParameters")

    def _select_signal_processing_mode(self, signals, simulation_params):
        """
        Selects the appropriate signal processing mode based on simulation parameters.

        Parameters:
        - signals: The input signals to process.
        - simulation_params: The simulation parameters affecting processing.

        Returns:
        Processed signals after applying the selected signal processing mode.
        """
        if simulation_params.signal_processing_mode in ['advanced', 'complex']:
            return self.advanced_signal_processing(signals, simulation_params)
        else:
            return self.basic_signal_processing(signals, simulation_params)
    
class ConnectionWithDelay:
    """
    Represents a connection between neurons with an optional delay and signal inversion.
    """

    def __init__(self, strength, repeat_factor, invert_signal):
        """
        Initializes the connection with specified parameters.

        Parameters:
        - strength: The strength of the connection.
        - repeat_factor: The factor by which the signal is repeated.
        - invert_signal: A boolean indicating whether the signal should be inverted.
        """
        self.strength = self._validate_param(strength, 'Strength', float, int, min_val=0, max_val=127)
        self.repeat_factor = self._validate_param(repeat_factor, 'Repeat Factor', int, min_val=1, max_val=5)
        self.invert_signal = self._validate_param(invert_signal, 'Invert Signal', bool)

        # Create a deque to store delayed signals
        self.delayed_signals = deque([0] * self.repeat_factor, maxlen=self.repeat_factor)

    def transmit(self, signal, simulation_params):
        """
        Transmits a signal through the connection, applying delay and inversion if needed.

        Parameters:
        - signal: The signal to transmit.
        - simulation_params: The simulation parameters affecting transmission.

        Returns:
        The delayed and possibly inverted signal.
        """
        if not isinstance(signal, (int, float)):
            raise TypeError("Signal must be a number")
        if not isinstance(simulation_params, SimulationParameters):
            raise TypeError("Simulation parameters must be an instance of SimulationParameters")

        if self.invert_signal:
            signal = -signal
        repeated_signal = signal * self.strength
        self.delayed_signals.append(repeated_signal)
        return self.delayed_signals.popleft()

    def _validate_param(self, param, name, *types, min_val=None, max_val=None):
        """
        Validates a parameter with type checking and range validation if applicable.

        Parameters:
        - param: The parameter to validate.
        - name: The name of the parameter (for error messages).
        - types: Allowed types for the parameter.
        - min_val: Optional minimum value for the parameter.
        - max_val: Optional maximum value for the parameter.

        Returns:
        The validated parameter.
        """
        if not isinstance(param, types):
            raise TypeError(f"{name} must be of type {types}")
        if min_val is not None and param < min_val:
            raise ValueError(f"{name} must be at least {min_val}")
        if max_val is not None and param > max_val:
            raise ValueError(f"{name} must not exceed {max_val}")
        return param
    
class MiniNetworkWithDelays:
    """
    Represents a mini neural network with neurons and connections, including delays and signal processing.
    """

    def __init__(self, neurons, connections):
        """
        Initializes the mini network with specified neurons and connections.

        Parameters:
        - neurons: A list of neurons in the network.
        - connections: A dictionary mapping tuples of neuron indices to connections.
        """
        if not all(isinstance(neuron, SimplifiedNeuronWithDelay) for neuron in neurons):
            raise TypeError("All neurons must be instances of SimplifiedNeuronWithDelay")
        if not isinstance(connections, dict):
            raise TypeError("Connections must be a dictionary")

        self.neurons = neurons
        self.connections = connections

    def simulate(self, external_input, current_time, simulation_params):
        """
        Simulates network activity based on external input, current time, and simulation parameters.

        Parameters:
        - external_input: The external input signals to the network.
        - current_time: The current time step in the simulation.
        - simulation_params: The simulation parameters affecting processing.

        Returns:
        A list of outputs from each neuron in the network.
        """
        if not isinstance(external_input, np.ndarray):
            raise TypeError("External input must be a NumPy array")
        if not isinstance(current_time, (int, float)):
            raise TypeError("Current time must be a number")
        if not isinstance(simulation_params, SimulationParameters):
            raise TypeError("Simulation parameters must be an instance of SimulationParameters")

        neuron_outputs = [0] * len(self.neurons)

        # Process external input for the first six neurons
        for i in range(6):
            neuron_input = external_input[i] if i < len(external_input) else np.zeros((simulation_params.neuron_signal_size,))
            neuron_outputs[i] = self.neurons[i].process_input(neuron_input.reshape(1, -1), current_time, simulation_params)

        # Process internal connections and time window for all neurons
        for i in range(len(self.neurons)):
            aggregated_input = np.zeros((simulation_params.neuron_signal_size,))
            for j in range(len(self.neurons)):
                if (j, i) in self.connections:
                    connection = self.connections[(j, i)]
                    transmitted_signal = connection.transmit(neuron_outputs[j], simulation_params)
                    transmitted_signal = np.array([transmitted_signal])  # Ensure transmitted_signal is a NumPy array
                    aggregated_input += transmitted_signal.reshape(-1)

            # Combine external and internal inputs for the first six neurons
            if i < 6:
                neuron_output_array = np.array([neuron_outputs[i]])
                aggregated_input += neuron_output_array.reshape(-1)

            # Process input for each neuron
            neuron_outputs[i] = self.neurons[i].process_input(aggregated_input.reshape(1, -1), current_time, simulation_params)

        return neuron_outputs
    
def calculate_total_neurons(N, W):
    """
    Calculates the total number of neurons in a layered neural network.

    Parameters:
    - N (int): Number of layers in the network.
    - W (int): Width of each layer.

    Returns:
    - int: Total number of neurons in the network.
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer")
    if not isinstance(W, int) or W <= 0:
        raise ValueError("W must be a positive integer")

    neurons_base_layer = W * 7 * (6 ** (N - 1))
    neurons_above_base = W * sum(6 ** (N - i) for i in range(2, N + 1))
    total_neurons = neurons_base_layer + neurons_above_base
    return total_neurons

def random_neuron_params(simulation_params):
    """
    Generates random parameters for a neuron based on simulation parameters.

    Parameters:
    - simulation_params: An instance of SimulationParameters containing global simulation settings.

    Returns:
    A dictionary of random neuron parameters.
    """
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError("simulation_params must be an instance of SimulationParameters")

    max_score_range = simulation_params.pattern_params.get('max_score_range', (10, 100))

    return {
        'threshold': np.clip(np.random.lognormal(mean=np.log(35), sigma=0.5), 0, 127),
        'over_threshold': np.clip(np.random.lognormal(mean=np.log(70), sigma=0.5), 0, 127),
        'base_signal_strength': np.clip(np.random.lognormal(mean=np.log(64), sigma=0.5), 1, 128),
        'refractory_period': np.clip(np.random.lognormal(mean=np.log(3), sigma=0.5), 1, 5),
        'spike_threshold': np.clip(np.random.lognormal(mean=np.log(50), sigma=0.5), 1, 127),
        'scaling_factor': np.random.uniform(0.1, 2.0),
        'neuron_state': random.choice(['excitatory', 'inhibitory', 'neutral']),
        'damping_factor': np.clip(np.random.lognormal(mean=np.log(2), sigma=0.3), 1, 5),
        'num_sub_windows': np.random.randint(1, 10),
        'amplification_factor': np.random.uniform(1, 5),
        'temporal_window_size': np.random.randint(5, 20),
        'pattern_params': {'max_score': np.random.uniform(*max_score_range)},
        'mapping_params': {
            'layer1_factor': np.random.uniform(0.1, 1.0),
            'layer2_exponent': np.random.uniform(1, 3),
            'steepness': np.random.uniform(0.1, 1.0)
        },
        'output_params': {'expansion_factor': np.random.uniform(1, 5)}
    }

def random_connection_params(simulation_params):
    """
    Generates random parameters for connections, influenced by simulation parameters.

    Parameters:
    - simulation_params: An instance of SimulationParameters containing global simulation settings.

    Returns:
    A dictionary of random connection parameters.
    """
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError("simulation_params must be an instance of SimulationParameters")

    return {
        'strength': np.clip(np.random.uniform(0.5, 1.0), 0, 127),
        'repeat_factor': np.random.randint(1, 5),
        'invert_signal': random.choice([True, False])
    }

def standard_input_transform(input_data, signal_size, num_neurons, simulation_params):
    """
    Transforms standard input based on the type of input_data.

    Parameters:
    - input_data: The input data to transform.
    - signal_size: The size of the signal for each neuron.
    - num_neurons: The number of neurons to which the input will be distributed.
    - simulation_params: An instance of SimulationParameters containing global simulation settings.

    Returns:
    Transformed input data suitable for processing by the neural network.
    """
    if not isinstance(signal_size, int) or signal_size <= 0:
        raise ValueError("signal_size must be a positive integer")
    if not isinstance(num_neurons, int) or num_neurons <= 0:
        raise ValueError("num_neurons must be a positive integer")
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError("simulation_params must be an instance of SimulationParameters")

    if isinstance(input_data, np.ndarray):
        return transform_array_input(input_data, signal_size, num_neurons)
    elif isinstance(input_data, str):
        return transform_text_input(input_data, signal_size, num_neurons)
    else:
        raise ValueError("Unsupported input type")

def transform_array_input(array_input, signal_size, num_neurons):
    """
    Transforms array input for neural network processing.

    Parameters:
    - array_input: The input array to transform.
    - signal_size: The size of the signal for each neuron.
    - num_neurons: The number of neurons to which the input will be distributed.

    Returns:
    A NumPy array reshaped to match the expected input dimensions for the neural network.
    """
    return np.reshape(array_input, (num_neurons, signal_size))

def transform_text_input(text_input, signal_size, num_neurons):
    """
    Transforms text input for neural network processing.

    Parameters:
    - text_input: The input text to transform.
    - signal_size: The size of the signal for each neuron.
    - num_neurons: The number of neurons to which the input will be distributed.

    Returns:
    A NumPy array representing the transformed text input, suitable for neural network processing.
    """
    return np.zeros((num_neurons, signal_size))

def dynamic_input_generator(num_time_steps, simulation_params, external_input=None):
    """
    Generates dynamic input for a given number of time steps.

    Parameters:
    - num_time_steps: The number of time steps for which to generate input.
    - simulation_params: An instance of SimulationParameters containing global simulation settings.
    - external_input: Optional external input to transform and use.

    Returns:
    A NumPy array containing the generated dynamic input for each time step.
    """
    if not isinstance(num_time_steps, int) or num_time_steps <= 0:
        raise ValueError("num_time_steps must be a positive integer")
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError("simulation_params must be an instance of SimulationParameters")

    num_neurons = simulation_params.input_layer_neuron_count
    signal_size = simulation_params.neuron_signal_size

    if external_input is not None:
        transformed_input = standard_input_transform(external_input, signal_size, num_neurons, simulation_params)
    else:
        transformed_input = np.random.randint(-128, 128, size=(num_neurons, signal_size))

    return np.array([transformed_input for _ in range(num_time_steps)])
    
def create_mini_network(simulation_params, connection_mode='specified'):
    """
    Creates a mini neural network based on the provided simulation parameters and connection mode.

    Parameters:
    - simulation_params: An instance of SimulationParameters containing global simulation settings.
    - connection_mode: A string indicating the mode for defining connections between neurons.

    Returns:
    An instance of MiniNetworkWithDelays representing the created mini neural network.
    """
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError("simulation_params must be an instance of SimulationParameters")
    
    neuron_params = [random_neuron_params(simulation_params) for _ in range(7)]
    neurons = [SimplifiedNeuronWithDelay(**params, simulation_params=simulation_params) for params in neuron_params]

    connections = {}
    if connection_mode == 'specified':
        specified_connections = [
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
            (1, 3), (3, 5), (5, 1), (2, 4), (4, 6), (6, 2),
            (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)
        ]
        for i, j in specified_connections:
            connections[(i, j)] = ConnectionWithDelay(**random_connection_params(simulation_params))
    elif connection_mode == 'other_mode':
        # Implement other connection modes as needed.
        pass
    # Additional connection modes can be added here.

    return MiniNetworkWithDelays(neurons, connections)

def generate_new_data(existing_data, num_new_points, buffer_size=100):
    """
    Generates new random data for the neural network simulation.

    Parameters:
    - existing_data (np.ndarray): Current data buffer of the simulation.
    - num_new_points (int): Number of new data points to generate.
    - buffer_size (int): Size of the data buffer to maintain.

    Returns:
    - np.ndarray: Updated data buffer with new random data appended.
    """
    new_data = np.random.rand(num_new_points, 7)
    
    combined_data = np.vstack((existing_data, new_data))
    
    if combined_data.shape[0] > buffer_size:
        return combined_data[-buffer_size:]
    else:
        return combined_data

def visualize_network_activity(initial_data):
    """
    Visualizes the neural network activity using Matplotlib.

    Parameters:
    - initial_data (np.ndarray): The initial dataset for the simulation.
    """
    global ani
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    initial_data = np.array(initial_data) if not isinstance(initial_data, np.ndarray) else initial_data
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    lines1 = [ax1.plot(initial_data[:, i], label=f'Neuron {i+1}', color=colors[i], linewidth=1)[0] for i in range(6)]
    lines2 = [ax2.plot(initial_data[:, 6], label='Node 7', color='k', linewidth=2)[0]]

    ax1.set_title('Activity of Neurons 1-6')
    ax2.set_title('Activity of Node 7')
    for ax in [ax1, ax2]:
        ax.legend()
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Activity')
        ax.grid(True)

    data_buffer = np.copy(initial_data)
    num_new_points = [1]  

    dynamic_x_axis = [True]  
    x_axis_range = [20]  

    ax_slider_speed = plt.axes([0.25, 0.01, 0.50, 0.03], facecolor='lightgoldenrodyellow')
    speed_slider = Slider(ax_slider_speed, 'Speed', 1, 100, valinit=50)

    ax_check = plt.axes([0.8, 0.02, 0.15, 0.1], facecolor='lightgoldenrodyellow')
    dynamic_check = CheckButtons(ax_check, ['Dynamic X-Axis'], [True])

    ax_button = plt.axes([0.05, 0.02, 0.1, 0.05])
    reset_button = Button(ax_button, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
    
    def update_x_range(val):
        nonlocal x_axis_range  
        x_axis_range[0] = int(val)
        if not dynamic_x_axis[0]:
            ax1.set_xlim(0, x_axis_range[0])
            ax2.set_xlim(0, x_axis_range[0])
            fig.canvas.draw_idle()

    def toggle_dynamic(label):
        dynamic_x_axis[0] = not dynamic_x_axis[0]

    def reset_to_default(event):
        dynamic_x_axis[0] = True
        x_axis_range[0] = 20
        dynamic_check.set_active(0)
    
    dynamic_check.on_clicked(toggle_dynamic)
    reset_button.on_clicked(reset_to_default)
    
    def update_speed(val):
        """
        Update the animation speed based on the slider value.
        """
        ani.event_source.interval = 1000 / val

    speed_slider.on_changed(update_speed)
    
    ani = FuncAnimation(fig, update_plot, fargs=(lines1 + lines2, data_buffer, num_new_points[0], ax1, ax2, 300, dynamic_x_axis[0], x_axis_range[0]), interval=100, blit=False)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()
    ani = None
    
def update_plot(frame, lines, data_buffer, num_new_points, ax1, ax2, buffer_size=300, dynamic_x_axis=True, x_axis_range=20):
    """
    Update function for the animation. This function is called for each frame in the animation.

    Parameters:
    - frame (int): Current frame number.
    - lines (list): List of Line2D objects representing the data lines on the plot.
    - data_buffer (np.ndarray): Buffer containing the data to be plotted.
    - num_new_points (int): Number of new data points generated for each frame.
    - ax1 (matplotlib.axes.Axes): Axes object for the first subplot.
    - ax2 (matplotlib.axes.Axes): Axes object for the second subplot.
    - buffer_size (int): Size of the data buffer.
    - dynamic_x_axis (bool): Flag to determine if the x-axis should be dynamic.
    - x_axis_range (int): Range of the x-axis.
    """
    try:
        new_data = generate_new_data(data_buffer[-min(len(data_buffer), buffer_size):], num_new_points)
        data_buffer = np.vstack((data_buffer, new_data))[-buffer_size:]

        x_data = np.arange(start=max(0, len(data_buffer) - buffer_size), stop=len(data_buffer))

        for i, line in enumerate(lines):
            line.set_data(x_data, data_buffer[-buffer_size:, i])

        if dynamic_x_axis:
            x_min_current = max(0, len(data_buffer) - buffer_size)
            x_max_current = len(data_buffer)
        else:
            x_min_current = 0
            x_max_current = x_axis_range

        ax1.set_xlim(x_min_current, x_max_current)
        ax2.set_xlim(x_min_current, x_max_current)

        adjust_y_axis_limits(ax1, data_buffer[-buffer_size:, :6])
        adjust_y_axis_limits(ax2, data_buffer[-buffer_size:, 6])

        return lines

    except Exception as e:
        logging.exception("Exception occurred during plot update")
        raise

def adjust_y_axis_limits(ax, data, margin_factor=0.1):
    """
    Adjusts the y-axis limits dynamically based on the range of the data currently being visualized.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to adjust.
    - data (np.ndarray): The data currently being visualized.
    - margin_factor (float): A factor to determine the margin around the data range for better visibility.
    """
    y_min, y_max = np.min(data), np.max(data)
    y_margin = (y_max - y_min) * margin_factor
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    logging.debug(f"Y-axis for ax set to {y_min - y_margin}, {y_max + y_margin}")
    
def main(time_steps=100):
    """
    Main function to run the neural network simulation.

    Parameters:
    - time_steps (int): Number of time steps for the simulation.
    """
    try:
        global ani  # Declare ani as global
        simulation_params = SimulationParameters()
        mini_network = create_mini_network(simulation_params)

        input_signals = dynamic_input_generator(time_steps, simulation_params)

        network_output = []
        for current_time in range(time_steps):
            output = mini_network.simulate(input_signals[current_time], current_time, simulation_params)
            network_output.append(output)

        ani = visualize_network_activity(network_output)  # Assign the result to ani once
    except Exception as e:
        logging.error(f"An error occurred during the simulation: {e}")
        print(f"An error occurred during the simulation: {e}")

if __name__ == "__main__":
    cProfile.run('main()')