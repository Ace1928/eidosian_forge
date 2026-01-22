
import numpy as np
import matplotlib.pyplot as plt

class EnhancedNeuron:
    # ... (Same as before)

class EnhancedConnection:
    # ... (Same as before)

class MiniNetworkWithFeedback:
    # ... (Same as before)

# Morse code sequence and signal creation
# ... (Same as before)

# Main simulation and visualization code
def main():
    # Morse code for each letter A-Z
    morse_code = {
        # ... (Same as before)
    }

    # Function to create input signals for a given letter
    # ... (Same as before)

    # Combine signals for all letters with an initial period of 0 inputs for 100 time steps
    combined_input_signals = [0] * 100
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        combined_input_signals.extend(create_input_for_letter(letter))
        combined_input_signals.extend([0] * letter_space)  # Space between letters

    # Extend the combined input signals to fit 7 neurons
    input_signals_morse = [combined_input_signals for _ in range(7)]

    # Creating the network and running a simulation
    global_scaling_factor = np.random.normal(1, 0.2)
    network = MiniNetworkWithFeedback(global_scaling_factor)
    outputs_morse = network.simulate(input_signals_morse)

    # Visualization code
    fig, axes = plt.subplots(8, 1, figsize=(15, 30))
    for i in range(7):
        axes[i].plot([output[i] for output in outputs_morse], label=f'Neuron {i+1}')
        axes[i].legend(loc="upper right")
        axes[i].set_ylabel("Output")
        axes[i].set_title(f"Output of Neuron {i+1}")

    axes[7].plot(combined_input_signals, label='Morse Code Input Sequence', color='black')
    axes[7].legend(loc="upper right")
    axes[7].set_ylabel("Amplitude")
    axes[7].set_title("Morse Code Input Sequence")

    axes[-1].set_xlabel("Time Steps")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
