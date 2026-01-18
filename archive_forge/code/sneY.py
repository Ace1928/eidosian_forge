# fractal_generator.py

import math
from config import global_config


class FractalGenerator:
    """
    Generates a fractal hexagonally structured neural network.
    """

    def __init__(self, base_layer_hexagons):
        self.base_layer_hexagons = base_layer_hexagons
        self.layers = self.calculate_layers()

    def calculate_layers(self):
        """
        Calculates the number of hexagons in each layer based on the fractal pattern.

        Returns:
            list: A list containing the number of hexagons in each layer.
        """
        layers = [self.base_layer_hexagons]
        current_hexagons = self.base_layer_hexagons

        # Calculate the number of hexagons in each subsequent layer
        # The pattern increases by a factor of the previous layer's hexagons + 6 (for the hexagonal growth)
        while current_hexagons > 1:
            current_hexagons = math.ceil(current_hexagons / 2) + 6
            layers.append(current_hexagons)

        return layers

    def generate_network(self):
        """
        Generates the neural network based on the calculated layers.

        Returns:
            dict: A dictionary representing the neural network with layers and hexagons.
        """
        network = {}
        for layer_index, hexagons in enumerate(self.layers):
            network[f"Layer_{layer_index + 1}"] = {
                "Hexagons": hexagons,
                "Activation_Function": global_config.get_activation_function(
                    global_config.default_activation_function
                ),
            }

        return network

    def visualize_network(self, network):
        """
        Placeholder for network visualization logic.

        Parameters:
            network (dict): The neural network to visualize.
        """
        # Visualization logic will be implemented here.
        pass


# Example usage
if __name__ == "__main__":
    fractal_generator = FractalGenerator(global_config.base_layer_hexagons)
    neural_network = fractal_generator.generate_network()
    print(neural_network)
    fractal_generator.visualize_network(neural_network)
