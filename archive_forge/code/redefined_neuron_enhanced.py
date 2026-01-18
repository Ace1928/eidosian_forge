
from lognormal_around import lognormal_around
import numpy as np
import random

class RedefinedNeuron:
    """
    Redefined Neuron class with stochastic firing, adaptive thresholds, and other specified parameters.
    """
    def __init__(self, id, global_scaling_factor=1.0, is_neuron_7=False):
        self.id = id
        self.base_signal_strength = lognormal_around(30, 20, 40) * global_scaling_factor
        self.initial_threshold = lognormal_around(60, 40, 80) * global_scaling_factor
        self.threshold = self.initial_threshold
        self.over_threshold = 1.5 * self.threshold
        self.excitotoxic_threshold = lognormal_around(3 * self.threshold, 2 * self.threshold, 4 * self.threshold)
        self.oscillation_amplitude = lognormal_around(0.5 * self.threshold, 0.4 * self.threshold, 0.6 * self.threshold)
        self.refractory_period = int(lognormal_around(5.5, 1, 10))
        self.global_scaling_factor = global_scaling_factor
        self.oscillation_base_frequency = np.random.uniform(18, 22)  # Frequency range for oscillations
        self.firing_history = []

        if is_neuron_7:
            self.threshold /= 2
            self.over_threshold *= 1.5
            self.excitotoxic_threshold *= 1.5

    def process_input_with_subthreshold_oscillation(self, input_signal, time_step, time_scale=0.01):
        """
        Processes input signal with subthreshold oscillation and stochastic firing.
        """
        oscillation = self.oscillation_amplitude * np.sin(2 * np.pi * self.oscillation_base_frequency * time_step * time_scale)
        modulated_input = input_signal + oscillation
        firing_probability = self.calculate_firing_probability(modulated_input)
        if random.random() < firing_probability:
            self.output = self.base_signal_strength
            self.feedback_signal = (modulated_input - self.threshold) / self.threshold
            # Update refractory period and threshold based on activity
            self.update_refractory_period(self.feedback_signal)
            self.update_threshold()
            self.firing_history.append(time_step)
        else:
            self.output = 0
            self.feedback_signal = 0

    def calculate_firing_probability(self, modulated_input):
        """
        Calculate the probability of neuron firing based on the input signal.
        """
        if modulated_input < self.threshold:
            return 0
        else:
            # Probability increases as the input signal gets stronger
            return min(1, (modulated_input - self.threshold) / self.threshold)

    def update_threshold(self):
        """
        Update the neuron's firing threshold based on recent firing history.
        """
        # Adaptive mechanism to adjust threshold based on recent activity
        if len(self.firing_history) > 5:  # Consider last 5 firings
            recent_firings = self.firing_history[-5:]
            if max(recent_firings) - min(recent_firings) < 20:  # If firings are close together
                self.threshold -= 0.1 * self.global_scaling_factor  # Lower threshold
            else:
                self.threshold = self.initial_threshold  # Reset to initial threshold
        self.threshold = max(20, self.threshold)  # Ensure threshold doesn't go too low

    def update_refractory_period(self, activity_level):
        """
        Update the refractory period based on neuron activity level.
        """
        # Simple rule to vary refractory period
        self.refractory_period = int(lognormal_around(5.5 + activity_level, 1, 10) * self.global_scaling_factor)
