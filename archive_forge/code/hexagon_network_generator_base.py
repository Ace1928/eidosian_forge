from typing import List, Tuple, Dict, Optional, Union, Any
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
import sys
import logging

# Setup logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define type aliases for clarity and type safety
Point3D = Tuple[float, float, float]
Hexagon3D = List[Point3D]
Structure3D = Dict[int, List[Hexagon3D]]

class Arrow3D(FancyArrowPatch):
    """
    Draws a 3D arrow using the FancyArrowPatch from matplotlib.

    Attributes:
        xs (List[float]): The x coordinates of the start and end points of the arrow.
        ys (List[float]): The y coordinates of the start and end points of the arrow.
        zs (List[float]): The z coordinates of the start and end points of the arrow.
    """
    def __init__(self, xs: List[float], ys: List[float], zs: List[float], *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer: Optional[Any] = None) -> float:
        """
        Projects the 3D arrow into a 2D representation using the renderer's projection method.
        """
        xs, ys, zs = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, renderer.M if renderer else plt.gca().get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
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
        self.layers = layers
        self.side_length = side_length
        self.structure = self._generate_3d_structure()

    def _generate_hexagon(self, center: Point3D, elevation: float) -> Hexagon3D:
        """
        Generates the vertices of a hexagon given its center and elevation.
        """
        vertices = []
        for i in range(6):
            angle_rad = 2 * math.pi / 6 * i
            x = center[0] + self.side_length * math.cos(angle_rad)
            y = center[1] + self.side_length * math.sin(angle_rad)
            vertices.append((x, y, elevation))
        return vertices

    def _generate_3d_structure(self) -> Structure3D:
        """
        Generates a 3D structure of stacked hexagons.
        """
        structure = {}
        elevation = 0.0
        elevation_step = self.side_length * math.sqrt(3) / 2
        for layer in range(self.layers):
            structure[layer] = [self._generate_hexagon((layer * self.side_length * 1.5, 0, layer * elevation_step), layer * elevation_step)]

        base_center = (0.0, 0.0, elevation)
        structure[0] = [self._generate_hexagon(base_center, elevation)]

        for layer in range(1, self.layers):
            elevation += elevation_step
            previous_layer_hexagons = structure[layer - 1]
            current_layer_hexagons = []

            for hexagon in previous_layer_hexagons:
                for i in range(6):
                    angle_rad = math.pi / 3 * i
                    x = hexagon[0][0] + self.side_length * math.cos(angle_rad) * 2
                    y = hexagon[0][1] + self.side_length * math.sin(angle_rad) * 2
                    new_hexagon_center = (x, y, elevation)
                    if not any(np.allclose(new_hexagon_center, h[0]) for h in current_layer_hexagons):
                        current_layer_hexagons.append(self._generate_hexagon(new_hexagon_center, elevation))
            structure[layer] = current_layer_hexagons

        return structure

    def plot_structure(self):
        """
        Plots the 3D hexagonal structure with an interactive matplotlib figure.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        color_map = plt.get_cmap('viridis')

        for layer, hexagons in self.structure.items():
            color = color_map(layer / self.layers)
            for hexagon in hexagons:
                self.hexagon_connections(hexagon, ax, color=color)
                xs, ys, zs = zip(*hexagon)
                ax.plot(xs, ys, zs, color=color)

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

