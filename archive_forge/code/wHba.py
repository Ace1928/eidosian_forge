```python
# visualization.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import cos, sin, pi

def draw_hexagon(ax, center, size):
    """
    Draws a single hexagon on the given Matplotlib axis.

    Parameters:
        ax (matplotlib.axes.Axes): The Matplotlib axis to draw on.
        center (tuple): The (x, y) coordinates of the hexagon's center.
        size (float): The radius of the hexagon.
    """
    for i in range(6):
        x1 = center[0] + size * cos(pi / 3 * i)
        y1 = center[1] + size * sin(pi / 3 * i)
        x2 = center[0] + size * cos(pi / 3 * (i + 1))
        y2 = center[1] + size * sin(pi / 3 * (i + 1))
        ax.add_line(plt.Line2D((x1, x2), (y1, y2), color='black'))

def visualize_network(network_structure):
    """
    Visualizes the fractal hexagonally structured neural network.

    Parameters:
        network_structure (dict): The structure of the neural network.
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_axis_off()

    layer_gap = 2  # Vertical gap between layers
    hexagon_size = 0.5  # Radius of each hexagon
    layer_y = 0  # Initial y-coordinate for the first layer

    for layer_name, layer_info in network_structure.items():
        hexagons = layer_info['Hexagons']
        layer_width = hexagons * 1.5 * hexagon_size  # Approximate width of the layer
        start_x = -layer_width / 2  # Starting x-coordinate for hexagons in this layer

        for i in range(hexagons):
            hexagon_x = start_x + i * 1.5 * hexagon_size
            draw_hexagon(ax, (hexagon_x, layer_y), hexagon_size)

        layer_y -= layer_gap  # Move down for the next layer

    plt.show()

# Example usage
if __name__ == "__main__":
    from fractal_generator import FractalGenerator
    from config import global_config

    fractal_generator = FractalGenerator(global_config.base_layer_hexagons)
    neural_network = fractal_generator.generate_network()
    visualize_network(neural_network)
```
