
# Instantiating neurons and connections with the redefined classes
neurons = {node_id: RedefinedNeuron(node_id, is_neuron_7=(node_id == 7)) for node_id in G.nodes}
connections = {edge: RedefinedConnection() for edge in G.edges}

# Simulation function for the neural network
def simulate_network(input_signals, time_steps):
    neuron_outputs = {node_id: np.zeros(time_steps) for node_id in G.nodes}
    feedback_signals = {node_id: np.zeros(time_steps) for node_id in G.nodes}

    for t in range(time_steps):
        for node_id in range(1, 8):
            neuron = neurons[node_id]
            input_signal = input_signals.get(node_id, [0] * time_steps)[t]
            neuron.process_input_with_subthreshold_oscillation(input_signal, t)
            neuron_outputs[node_id][t] = neuron.output
            feedback_signals[node_id][t] = neuron.feedback_signal

        # Propagating signals with connection delays
        for (source, target), connection in connections.items():
            delay = connection.delay
            propagated_time = t - delay
            if propagated_time >= 0:
                propagated_signal = neuron_outputs[source][propagated_time] * connection.strength
                neurons[target].process_input_with_subthreshold_oscillation(propagated_signal, t)

    return neuron_outputs, feedback_signals

# Example usage of the simulation
if __name__ == '__main__':
    # Creating a sine wave input signal
    time_scale = np.linspace(0, 1, 100)  # 1 second divided into 100 time steps
    sine_wave_input = 50 * np.sin(2 * np.pi * 20 * time_scale)  # 20 Hz frequency, amplitude 50

    # Running the simulation
    input_signals = {node_id: sine_wave_input for node_id in range(1, 7)}  # Same input to all input nodes
    outputs, feedbacks = simulate_network(input_signals, 100)

    # Visualizing the results (requires matplotlib)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 18))
    for i, node_id in enumerate(G.nodes):
        plt.subplot(len(G.nodes) + 1, 1, i + 1)
        plt.plot(outputs[node_id], label=f'Output of Neuron {node_id}')
        plt.plot(feedbacks[node_id], label=f'Feedback from Neuron {node_id}', linestyle='--')
        plt.ylabel('Signal Strength')
        plt.legend(loc='upper right')
        plt.title(f'Neuron {node_id} Activity')
    plt.subplot(len(G.nodes) + 1, 1, len(G.nodes) + 1)
    plt.plot(sine_wave_input, label='Sine Wave Input Signal', color='black')
    plt.ylabel('Amplitude')
    plt.xlabel('Time Steps')
    plt.title('Sine Wave Input Signal')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
