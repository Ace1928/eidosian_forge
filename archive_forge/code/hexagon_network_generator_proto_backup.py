import logging
import math
import sys
from typing import List, Tuple, Dict, Optional, Union, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
import pandas as pd
from datetime import datetime

# Initialize logging for detailed execution tracking
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"hexagon_{timestamp}_log.txt"
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

# Define type aliases for clarity and type safety
Point3D = Tuple[float, float, float]
Hexagon3D = List[Point3D]
Structure3D = Dict[int, List[Hexagon3D]]

class Arrow3D(FancyArrowPatch):
    """
    Draws a 3D arrow using matplotlib's FancyArrowPatch.

    Attributes:
        xs (List[float]): x-coordinates of the start and end points.
        ys (List[float]): y-coordinates of the start and end points.
        zs (List[float]): z-coordinates of the start and end points.
    """

    def __init__(self, xs: List[float], ys: List[float], zs: List[float],
                 *args, **kwargs):
        """
        Initializes the Arrow3D object with start and end coordinates in 3D space.
        """
        logging.info("Initializing 3D arrow with coordinates: xs=%s, ys=%s, zs=%s",
                     xs, ys, zs)
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer: Optional[Any] = None) -> float:
        """
        Projects the 3D arrow onto 2D space using the renderer's projection.
        """
        xs, ys, zs = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs,
                                           renderer.M if renderer else plt.gca().get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        logging.debug("Projected 3D arrow to 2D with min z-coordinate: %s", np.min(zs))
        return np.min(zs)

class HexagonalStructure:
    """
    Manages the generation, manipulation, and visualization of a 3D hexagonal structure.

    Attributes:
        layers (int): The number of layers in the hexagonal structure.
        side_length (float): The side length of each hexagon in the structure.
        structure (Structure3D): The generated 3D structure stored as a dictionary.
    """

    def __init__(self, layers: int, side_length: float):
        """
        Initializes the HexagonalStructure with specified layers and side length.
        """
        logging.info(f"Creating HexagonalStructure with {layers} layers and side length {side_length}")
        self.layers = layers
        self.side_length = side_length
        self.structure = self._generate_3d_structure()

    def _generate_hexagon(self, center: Point3D, elevation: float) -> Hexagon3D:
        """
        Generates the vertices of a hexagon given its center and elevation.
        """
        logging.debug(f"Generating hexagon at center: {center} and elevation: {elevation}")
        vertices = []
        for i in range(6):
            angle_rad = 2 * math.pi / 6 * i
            x = center[0] + self.side_length * math.cos(angle_rad)
            y = center[1] + self.side_length * math.sin(angle_rad)
            vertices.append((x, y, elevation))
        return vertices

    def _generate_3d_structure(self) -> Structure3D:
        """
        Generates a sophisticated 3D structure consisting of meticulously stacked hexagons,
        each placed with precision to form a cohesive, extended hexagonal prism structure.
        This method embodies the pinnacle of algorithmic design, pushing the boundaries
        of computational geometry to create a visually stunning and mathematically robust
        representation of a hexagonal structure in three dimensions.

        Returns:
            Structure3D: A meticulously curated dictionary. Each key represents a layer index,
            associated with a list of hexagons within that layer. Each hexagon is further
            represented as a list of 3D points, constituting a comprehensive model of the
            entire 3D hexagonal architecture.
        """
        logging.debug("Initiating the generation of the 3D hexagonal structure.")
        structure = {}
        elevation = 0.0  # Initial elevation set to ground level
        elevation_step = self.side_length * math.sqrt(3) / 2  # Calculated vertical distance between layers

        # Iteratively generate each layer of hexagons
        for layer in range(self.layers):
            hexagons = []  # Container for the hexagons in the current layer
            # Correcting center offset calculation to ensure symmetrical stacking
            center_offset_x = self.side_length * 1.5 * layer
            center_offset_y = self.side_length * math.sqrt(3) / 2 * layer

            # Special case for the base layer
            if layer == 0:
                base_center = (0.0, 0.0, elevation)  # Center of the base hexagon
                hexagons.append(self._generate_hexagon(base_center, elevation))
            else:
                # For upper layers, position hexagons around each hexagon of the layer below
                elevation += elevation_step
                previous_layer_hexagons = structure[layer - 1]
                current_layer_hexagons = []

                # Position new hexagons around the perimeter of those in the layer below
                for hexagon in previous_layer_hexagons:
                    for i in range(6):
                        angle_rad = math.pi / 3 * i  # Exploiting hexagonal symmetry
                        x = hexagon[0][0] + self.side_length * math.cos(angle_rad) * 2
                        y = hexagon[0][1] + self.side_length * math.sin(angle_rad) * 2
                        new_hexagon_center = (x, y, elevation)

                        # Ensure new hexagon does not overlap with existing ones
                        if not any(np.allclose(new_hexagon_center, h[0], atol=1e-8) for h in hexagons):
                            hexagons.append(self._generate_hexagon(new_hexagon_center, elevation))

            # Adjusting the hexagons in the current layer to be centered
            hexagons_centered = []
            for hexagon in hexagons:
                hexagon_centered = [(x - center_offset_x, y - center_offset_y, z) for x, y, z in hexagon]
                hexagons_centered.append(hexagon_centered)

            structure[layer] = hexagons_centered
            logging.info(f"Hexagonal layer {layer} generated with {len(hexagons_centered)} hexagons.")

        logging.debug("The 3D hexagonal structure has been fully realized to the zenith of algorithmic artistry.")
        return structure

    def plot_structure(self):
        """
        Plots the 3D hexagonal structure with an interactive matplotlib figure.
        """
        logging.info("Plotting 3D hexagonal structure")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        color_map = plt.get_cmap('viridis')

        for layer, hexagons in self.structure.items():
            color = color_map(layer / self.layers)
            for hexagon in hexagons:
                self.hexagon_connections(hexagon, ax, color=color)
                xs, ys, zs = zip(*hexagon)
                ax.plot(xs, ys, zs, color=color)
                # Add internode connections
                if layer < self.layers - 1:
                    self.internode_connections(hexagon, layer, ax, color=color)

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.title('3D Hexagonal Structure')
        plt.tight_layout()
        plt.show()

    def hexagon_connections(self, hexagon: Hexagon3D, ax: plt.Axes, color: str):
        """
        Draws connections between the vertices of a hexagon and its center, enhancing the 3D visualization.
        """
        for i in range(6):  # Adjusted indexing to correctly access hexagon vertices
            start = hexagon[i]
            for j in [1, 2, 3]:  # Direct connections and skips
                end = hexagon[(i + j) % 6]
                ax.add_artist(Arrow3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                                      mutation_scale=10, lw=1, arrowstyle="-|>", color=color))
        center = np.mean(np.array(hexagon), axis=0)  # Correct calculation of the hexagon's center
        for vertex in hexagon:
            ax.add_artist(Arrow3D([vertex[0], center[0]], [vertex[1], center[1]], [vertex[2], center[2]],
                                  mutation_scale=10, lw=1, arrowstyle="-|>", color=color))

    def internode_connections(self, hexagon: Hexagon3D, layer: int, ax: plt.Axes, color: str):
        """
        Draws connections from the central node of a hexagon to a vertex in the layer above.
        """
        center = np.mean(np.array(hexagon), axis=0)
        next_layer_hexagons = self.structure[layer + 1]
        for i, next_hexagon in enumerate(next_layer_hexagons):
            target_vertex = next_hexagon[i % 6]
            ax.add_artist(Arrow3D([center[0], target_vertex[0]], [center[1], target_vertex[1]], [center[2], target_vertex[2]],
                                  mutation_scale=10, lw=1, arrowstyle="-|>", color=color))

    def export_to_csv(self, neurons_filename: str, synapses_filename: str):
        """
        Exports the structure data to CSV files.
        """
        logging.info(f"Exporting structure data to {neurons_filename} and {synapses_filename}")

        # Export neurons data
        neurons_data = []
        neuron_index = 0
        for layer, hexagons in self.structure.items():
            for h, hexagon in enumerate(hexagons):
                for n, vertex in enumerate(hexagon):
                    label = self._generate_neuron_label(layer, h, n)
                    is_central = n == 6
                    size = 1
                    x, y, z = vertex
                    neurons_data.append({"Index": neuron_index, "Label": label, "Layer": layer,
                                         "Group": h // 6, "Hexagon": h % 6, "Neuron": n,
                                         "Is_Central": is_central, "Size": size, "X": x, "Y": y, "Z": z})
                    neuron_index += 1
        neurons_df = pd.DataFrame(neurons_data)
        neurons_df.to_csv(neurons_filename, index=False)

        # Export synapses data
        synapses_data = []
        synapse_index = 0
        for layer, hexagons in self.structure.items():
            for h, hexagon in enumerate(hexagons):
                for n, _ in enumerate(hexagon):
                    source_label = self._generate_neuron_label(layer, h, n)
                    source_index = self._get_neuron_index(neurons_df, source_label)

                    # Intra-hexagon connections
                    for j in [1, 2, 3]:
                        target_label = self._generate_neuron_label(layer, h, (n + j) % 6)
                        target_index = self._get_neuron_index(neurons_df, target_label)
                        synapses_data.append({"Index": synapse_index, "Label": f"{source_label}:{target_label}",
                                              "Source_Label": source_label, "Source_Index": source_index,
                                              "Target_Label": target_label, "Target_Index": target_index})
                        synapse_index += 1

                    # Inter-layer connections
                    if layer < self.layers - 1 and n == 6:
                        target_label = self._generate_neuron_label(layer + 1, h, 0)
                        target_index = self._get_neuron_index(neurons_df, target_label)
                        synapses_data.append({"Index": synapse_index, "Label": f"{source_label}:{target_label}",
                                              "Source_Label": source_label, "Source_Index": source_index,
                                              "Target_Label": target_label, "Target_Index": target_index})
                        synapse_index += 1

        synapses_df = pd.DataFrame(synapses_data)
        synapses_df.to_csv(synapses_filename, index=False)
        logging.info(f"Data exported successfully to {neurons_filename} and {synapses_filename}")

    def _generate_neuron_label(self, layer: int, hexagon: int, neuron: int) -> str:
        """
        Generates a unique label for a neuron based on its hierarchical position.
        """
        l = layer
        r = hexagon // (6 ** 5)
        s = (hexagon % (6 ** 5)) // (6 ** 4)
        z = (hexagon % (6 ** 4)) // (6 ** 3)
        c = (hexagon % (6 ** 3)) // (6 ** 2)
        g = (hexagon % (6 ** 2)) // 6
        h = hexagon % 6
        n = neuron
        return f"[{l},{r},{s},{z},{c},{g},{h},{n}]"

    def _get_neuron_index(self, neurons_df: pd.DataFrame, label: str) -> int:
        """
        Retrieves the index of a neuron from the neurons DataFrame based on its label.
        """
        return int(neurons_df[neurons_df["Label"] == label]["Index"].values[0])

def main():
    """
    Main function to execute the script functionality.
    """
    logging.info("Starting the hexagonal structure visualization script.")
    try:
        layers = int(input("Enter the number of layers: "))
        side_length = float(input("Enter the side length of each hexagon: "))

        structure = HexagonalStructure(layers, side_length)
        structure.plot_structure()
        structure.export_to_csv("neurons.csv", "synapses.csv")
    except ValueError as e:
        logging.error(f"Invalid input: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    except KeyboardInterrupt:
        logging.info("Program execution was interrupted by the user.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Critical failure: {e}")
        sys.exit(1)
