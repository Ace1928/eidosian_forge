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
class MiniNetworkWithDelays:

    def __init__(self, neurons, connections):
        if not all((isinstance(neuron, SimplifiedNeuronWithDelay) for neuron in neurons)):
            raise TypeError('All neurons must be instances of SimplifiedNeuronWithDelay')
        if not isinstance(connections, dict):
            raise TypeError('Connections must be a dictionary')
        self.neurons = neurons
        self.connections = connections

    def simulate(self, external_input, current_time, simulation_params):
        if not isinstance(external_input, np.ndarray):
            raise TypeError('External input must be a NumPy array')
        if not isinstance(current_time, (int, float)):
            raise TypeError('Current time must be a number')
        if not isinstance(simulation_params, SimulationParameters):
            raise TypeError('Simulation parameters must be an instance of SimulationParameters')
        neuron_outputs = [0] * len(self.neurons)
        for i in range(6):
            neuron_input = external_input[i] if i < len(external_input) else np.zeros((simulation_params.neuron_signal_size,))
            neuron_outputs[i] = self.neurons[i].process_input(neuron_input.reshape(1, -1), current_time, simulation_params)
        for i in range(len(self.neurons)):
            aggregated_input = np.zeros((simulation_params.neuron_signal_size,))
            for j in range(len(self.neurons)):
                if (j, i) in self.connections:
                    connection = self.connections[j, i]
                    transmitted_signal = connection.transmit(neuron_outputs[j], simulation_params)
                    transmitted_signal = np.array([transmitted_signal])
                    aggregated_input += transmitted_signal.reshape(-1)
            if i < 6:
                neuron_output_array = np.array([neuron_outputs[i]])
                aggregated_input += neuron_output_array.reshape(-1)
            neuron_outputs[i] = self.neurons[i].process_input(aggregated_input.reshape(1, -1), current_time, simulation_params)
        return neuron_outputs