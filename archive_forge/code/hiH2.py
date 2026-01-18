import numpy as np
import matplotlib.pyplot as plt
import ctypes
import sys
import argparse
import random

plt.ion()  # Turn on interactive mode


# Neuron class with sparse firing and delay mechanism
class SimplifiedNeuronWithDelay:
    def __init__(
        self,
        threshold,
        base_signal_strength,
        refractory_period,
        scaling_factor,
        is_inhibitory,
        damping_factor,
    ):
        self.threshold = np.clip(threshold, 0, 127).astype(int)
        self.base_signal_strength = np.clip(base_signal_strength, -128, 127).astype(int)
        self.refractory_period = np.clip(refractory_period, 1, 5).astype(int)
        self.refractory_timer = 0
        self.scaling_factor = scaling_factor
        self.is_inhibitory = is_inhibitory
        self.damping_factor = damping_factor

    def process_input(self, input_signals):
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            return 0  # Neuron is in refractory period

        total_signal = np.sum(input_signals)
        scaled_signal = total_signal * self.scaling_factor
        damped_signal = scaled_signal // self.damping_factor

        if damped_signal >= self.threshold:
            self.refractory_timer = self.refractory_period
            return (
                self.base_signal_strength
                if not self.is_inhibitory
                else -self.base_signal_strength
            )
        return 0  # Below threshold


# Connection class with delay mechanism
class ConnectionWithDelay:
    def __init__(self, strength, repeat_factor):
        self.strength = np.clip(strength, 0, 127).astype(int)
        self.repeat_factor = np.clip(repeat_factor, 1, 5).astype(int)
        self.delayed_signals = [0] * self.repeat_factor

    def transmit(self, signal):
        repeated_signal = signal * self.strength
        self.delayed_signals.append(repeated_signal)
        transmitted_signal = self.delayed_signals.pop(0)
        return transmitted_signal


# Mini-network class with delays
class MiniNetworkWithDelays:
    def __init__(self, neurons, connections):
        self.neurons = neurons
        self.connections = connections

    def simulate(self, input_signals):
        outputs = []
        for input_signal in input_signals:  # Ensure input_signal is a list or array
            print(f"Simulating with input_signal: {input_signal}")  # Debugging print
            neuron_outputs = []
            for i, neuron in enumerate(self.neurons):
                # Use modulo operation to prevent out-of-bounds access
                index = i % len(input_signal)  # input_signal should be iterable
                neuron_output = neuron.process_input(input_signal[index])
                neuron_outputs.append(neuron_output)
            outputs.append(neuron_outputs)
        return outputs


def get_simulation_parameters():
    print("\nSimulation Parameter Setup:")

    num_layers = int(input("Enter number of layers (e.g., 3): "))
    num_networks_per_layer = int(
        input("Enter number of networks per layer (e.g., 6): ")
    )
    num_time_steps = int(
        input("Enter number of time steps for the simulation (e.g., 50): ")
    )
    num_networks_to_view = int(
        input(
            "Enter how many mini-networks you would like to view for each layer (e.g., 2): "
        )
    )

    return num_layers, num_networks_per_layer, num_time_steps, num_networks_to_view


def get_user_input():
    print("\nMain Menu:")
    print("1 - Start or Restart Simulation")
    print("2 - Alter Parameters")
    print("3 - Exit")
    return input("Enter your choice (1-3): ")


# Generate dynamic input signals
def dynamic_input_generator(num_time_steps, signal_size=6):
    return [np.random.randint(-128, 128, signal_size) for _ in range(num_time_steps)]


# Create a single mini-network
def create_mini_network(neuron_params, connection_params):
    neurons = [SimplifiedNeuronWithDelay(**params) for params in neuron_params]
    connections = [ConnectionWithDelay(**params) for params in connection_params]
    return MiniNetworkWithDelays(neurons, connections)


# Function to process a layer of mini-networks
def process_layer(input_signals, neuron_params, connection_params):
    mini_network = create_mini_network(neuron_params, connection_params)
    return mini_network.simulate(
        input_signals
    )  # Ensure input_signals is a list of lists or arrays


# Simulate network
def simulate_network(num_layers, num_networks_per_layer, num_time_steps):
    layer_outputs = []
    input_signals = dynamic_input_generator(
        num_time_steps
    )  # This should generate a list of lists or arrays

    for layer in range(num_layers):
        is_bottom_layer = layer == 0

        # Define neuron and connection parameters
        neuron_params = [
            {
                "threshold": 50,
                "base_signal_strength": 10,
                "refractory_period": 2,
                "scaling_factor": 1,
                "is_inhibitory": False,
                "damping_factor": 1,
            }
            for _ in range(7)
        ]
        connection_params = [{"strength": 127, "repeat_factor": 1} for _ in range(7)]

        network_outputs = [
            process_layer(
                (
                    input_signals
                    if is_bottom_layer
                    else [np.sum(signal) for signal in zip(*input_signals)]
                ),
                neuron_params,
                connection_params,
            )
            for _ in range(num_networks_per_layer)
        ]

        layer_outputs.append(network_outputs)
        # In simulate_network function
        if not is_bottom_layer:
            input_signals = [
                np.array([output for output in zip(*network_output)])
                for network_output in network_outputs
            ]
    return layer_outputs


# Visualize output with specific styling
def visualize_output(layer_outputs, num_networks_to_view):
    contrasting_colors = ["b", "g", "r", "c", "m", "y"]
    central_node_color = "k"  # Black for central node
    central_node_weight = 1.0
    other_node_weight = 0.25
    for layer_idx, layer_output in enumerate(layer_outputs):
        if len(layer_output) >= num_networks_to_view:
            for network_idx in range(num_networks_to_view):
                plt.figure()
                network_output = layer_output[network_idx]
                for node_idx, node_output in enumerate(zip(*network_output)):
                    color = (
                        central_node_color
                        if node_idx == 6
                        else contrasting_colors[node_idx % len(contrasting_colors)]
                    )
                    linewidth = (
                        central_node_weight if node_idx == 6 else other_node_weight
                    )
                    plt.plot(node_output, color=color, linewidth=linewidth)
                plt.title(f"Layer {layer_idx + 1}, Mini-Network {network_idx + 1}")
                plt.xlabel("Time Steps")
                plt.ylabel("Output Signal")
                plt.show()


def main():
    # Default simulation parameters
    num_layers, num_networks_per_layer, num_time_steps = 3, 6, 50
    num_networks_to_view = 2
    while True:
        user_choice = get_user_input()
        if user_choice == "1":
            print("\nStarting Simulation...")
            layer_outputs = simulate_network(
                num_layers, num_networks_per_layer, num_time_steps
            )
            visualize_output(layer_outputs, num_networks_to_view)
        elif user_choice == "2":
            print("\nAltering Parameters...")
            num_layers, num_networks_per_layer, num_time_steps, num_networks_to_view = (
                get_simulation_parameters()
            )
        elif user_choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")


if __name__ == "__main__":
    plt.ion()  # Turn on interactive mode
    main()
