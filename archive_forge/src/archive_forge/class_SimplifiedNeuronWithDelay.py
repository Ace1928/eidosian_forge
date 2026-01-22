import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from collections import deque
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
from matplotlib.animation import FuncAnimation
import logging
import datetime
import sys
import cProfile
class SimplifiedNeuronWithDelay:

    @staticmethod
    def generate_random_param(mean, sigma):
        return np.clip(np.random.normal(mean, sigma), -128, 127)

    def __init__(self, threshold, over_threshold, base_signal_strength, refractory_period, spike_threshold, scaling_factor, neuron_state, damping_factor, num_sub_windows, amplification_factor, temporal_window_size, pattern_params, mapping_params, output_params, simulation_params):
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
        self.input_history = deque(maxlen=temporal_window_size)
        self.last_spike_time = -1
        self.signal_processing_mode = simulation_params.signal_processing_mode
        self.pattern_params = pattern_params
        self.mapping_params = mapping_params
        self.output_params = output_params
        self.spike_magnitude = self.generate_random_param(30, 5)
        self.pattern_detection_threshold = self.generate_random_param(0.5, 0.1)
        self.mapping_steepness = self.generate_random_param(0.1, 0.02)
        self.mapping_skew_factor = self.generate_random_param(0.1, 0.02)
        self.pattern_params = self._validate_param(pattern_params, 'Pattern Params', dict)
        self.mapping_params = self._validate_param(mapping_params, 'Mapping Params', dict)
        self.output_params = self._validate_param(output_params, 'Output Params', dict)

    def _validate_param(self, param, name, *types, min_val=None, max_val=None):
        if not isinstance(param, types):
            raise TypeError(f'{name} must be of type {types}')
        if min_val is not None and param < min_val:
            raise ValueError(f'{name} must be at least {min_val}')
        if max_val is not None and param > max_val:
            raise ValueError(f'{name} must not exceed {max_val}')
        return param

    def _validate_choice(self, choice, name, valid_choices):
        if choice not in valid_choices:
            raise ValueError(f'{name} must be one of {valid_choices}')
        return choice

    def advanced_signal_processing(self, signals, simulation_params):
        self._validate_signals(signals)
        return signals * 2

    def basic_signal_processing(self, signals, simulation_params):
        self._validate_signals(signals)
        return signals * simulation_params.signal_amplification

    def spike_response(self):
        return self.spike_magnitude

    def detect_pattern(self, input_history, processing_params):
        if not isinstance(processing_params, dict):
            raise TypeError('Processing parameters must be a dictionary')
        pattern_strength = self.calculate_pattern_strength(input_history, processing_params)
        return pattern_strength > self.pattern_detection_threshold * processing_params['pattern_detection_sensitivity']

    def calculate_pattern_strength(self, input_history, processing_params):
        if len(input_history) == 0:
            return 0
        input_history_array = np.array(input_history)
        fft_result = np.fft.fft(input_history_array, axis=0)
        dominant_frequencies = np.abs(fft_result).mean(axis=0)
        entropy = -np.sum(dominant_frequencies * np.log(dominant_frequencies + 1e-09), axis=0)
        combined_score = np.sum(dominant_frequencies) + entropy
        normalized_score = np.clip(combined_score / processing_params['pattern_params']['max_score'], 0, 1)
        if isinstance(normalized_score, np.ndarray):
            normalized_score = int(np.mean(normalized_score))
        else:
            normalized_score = int(normalized_score)
        return normalized_score

    def state_mapping_function(self, input_signal_sum, mapping_params):
        default_steepness = 1.0
        default_skew_factor = 1.0
        steepness = self.mapping_steepness * mapping_params.get('steepness', default_steepness)
        skew_factor = self.mapping_skew_factor * mapping_params.get('skew_factor', default_skew_factor)
        sigmoid = 1 / (1 + np.exp(-steepness * (input_signal_sum - self.threshold)))
        skewed_value = np.log1p(abs(input_signal_sum)) ** skew_factor
        combined_value = sigmoid * skewed_value
        mapped_state = np.clip(int(combined_value * 6), -6, 6)
        return mapped_state

    def granular_output(self, state, output_params):
        expanded_output = state * output_params['expansion_factor']
        return expanded_output

    def process_input(self, input_signals, current_time, simulation_params):
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
        processing_params = {'pattern_detection_sensitivity': simulation_params.pattern_detection_sensitivity, 'pattern_params': self.pattern_params}
        if self.detect_pattern(self.input_history, processing_params):
            damped_signal *= 1.2
        mapped_state = self.state_mapping_function(damped_signal, self.mapping_params)
        output = mapped_state * np.sign(total_signal) * self.amplification_factor
        return self.granular_output(output, self.output_params)

    def _validate_signals(self, signals):
        if not isinstance(signals, np.ndarray):
            raise TypeError('Signals must be a NumPy array')
        if signals.ndim != 2:
            raise ValueError('Signals array must be 2-dimensional')

    def _validate_process_input_params(self, input_signals, current_time, simulation_params):
        self._validate_signals(input_signals)
        if not isinstance(current_time, (int, float)):
            raise TypeError('Current time must be a number')
        if not isinstance(simulation_params, SimulationParameters):
            raise TypeError('Simulation parameters must be an instance of SimulationParameters')

    def _select_signal_processing_mode(self, signals, simulation_params):
        if simulation_params.signal_processing_mode in ['advanced', 'complex']:
            return self.advanced_signal_processing(signals, simulation_params)
        else:
            return self.basic_signal_processing(signals, simulation_params)