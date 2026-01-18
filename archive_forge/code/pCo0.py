import pygame  # Importing the pygame library to handle game-specific functionalities.
import random  # Importing the random library to facilitate random number generation.
import heapq  # Importing the heapq library to provide an implementation of the heap queue algorithm.
import logging  # Importing the logging library to enable logging of messages of varying severity.
import numpy as np  # Importing the numpy library as np to provide support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
import networkx as nx  # Importing the networkx library as nx to create, manipulate, and study the structure, dynamics, and functions of complex networks.
from collections import (
    defaultdict,
)  # Importing defaultdict from collections to provide a dictionary with default values for missing keys.
from typing import (
    List,
    Tuple,
    Dict,
    Set,
)  # Importing specific types from the typing module to support type hinting.

# Initialize logging with maximum verbosity and detail
logging.basicConfig(
    level=logging.DEBUG,  # Setting the logging level to DEBUG to capture all levels of log messages.
    format="%(asctime)s - %(levelname)s - %(message)s",  # Defining the format of log messages to include the time, the level of severity, and the message.
)


def create_grid(size: int) -> np.ndarray:
    """
    Constructs a two-dimensional square grid of a specified size where each cell within the grid is initialized to zero. This grid is represented as a NumPy array, which is chosen for its optimized operations and superior performance characteristics, especially beneficial for handling large grids efficiently.

    The function employs the numpy.zeros method to instantiate the grid, ensuring that each cell is initialized to an integer value of zero. This initialization is crucial for the subsequent operations that may rely on a clean, zero-initialized state.

    Args:
        size (int): The dimension of the grid, which is used to define both the number of rows and the number of columns, given the grid is square in shape.

    Returns:
        np.ndarray: A 2D NumPy array with each element initialized to zero. This array provides a structured representation of the grid, encapsulating its complete structure in a format that is both accessible and efficient for computational operations.

    Raises:
        ValueError: If the provided size is less than 1, as a grid of zero or negative dimensions cannot be created.

    Detailed Description:
        - The function begins by validating the input size to ensure it is a positive integer. This is crucial to prevent the creation of an invalid grid which could lead to errors in downstream processes.
        - Upon successful validation, the numpy.zeros function is called with the appropriate dimensions and data type. This function is highly optimized for creating large arrays and is ideal for this purpose.
        - A debug log statement records the creation of the grid, noting its size and the initialization state. This log is vital for debugging and verifying the correct operation of the function in a development or troubleshooting scenario.
    """
    if size < 1:
        logging.error("Attempted to create a grid with non-positive dimension size.")
        raise ValueError("Size must be a positive integer greater than zero.")

    grid = np.zeros((size, size), dtype=int)
    logging.debug(
        f"Grid of size {size}x{size} created with all elements initialized to zero using NumPy."
    )

    return grid


def convert_to_graph(grid: np.ndarray) -> nx.Graph:
    """
    Converts a two-dimensional grid into a graph representation utilizing the NetworkX library. This library is chosen for its comprehensive capabilities and optimized performance for complex graph operations. Each cell within the grid is meticulously treated as a distinct node within the graph. These nodes are interconnected to their adjacent nodes (specifically in the up, down, left, and right directions) with edges. The weights of these edges are assigned using a random number generation mechanism to ensure variability and complexity in the graph structure.

    Args:
        grid (np.ndarray): A two-dimensional NumPy array representing the grid, where each element corresponds to a potential node in the graph.

    Returns:
        nx.Graph: A meticulously constructed NetworkX graph where each node corresponds to a cell in the grid. Edges connect each node to its adjacent nodes, with weights assigned randomly to each edge to enhance the complexity and utility of the graph.
    """
    # Determine the size of the grid based on its first dimension
    size = grid.shape[0]
    # Initialize a new graph object using NetworkX to ensure optimal graph manipulation capabilities
    graph = nx.Graph()
    # Iterate over each cell in the grid to convert it into a node in the graph
    for i in range(size):
        for j in range(size):
            # Define the potential movements to adjacent cells: up, down, left, right
            adjacent_movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            # Process each potential movement to determine valid graph edges
            for delta_i, delta_j in adjacent_movements:
                # Calculate the new position based on the current position and the movement delta
                new_i, new_j = i + delta_i, j + delta_j
                # Ensure the new position is within the bounds of the grid to maintain validity
                if 0 <= new_i < size and 0 <= new_j < size:
                    # Generate a random weight for the edge to ensure complexity and variability in the graph
                    weight = random.random()
                    # Add an edge to the graph between the current node and the adjacent node with the calculated random weight
                    graph.add_edge((i, j), (new_i, new_j), weight=weight)
                    # Log the addition of each edge with detailed debugging information to ensure traceability and transparency
                    logging.debug(
                        f"Edge added from {(i, j)} to {(new_i, new_j)} with weight {weight} using NetworkX."
                    )
    # Return the fully constructed graph, ensuring that all nodes and edges are included as per the original grid structure
    return graph


def draw_path(
    screen: pygame.Surface,
    path: List[Tuple[int, int]],
    cell_size: int,
    current_index: int,
    max_length: int,
    frame_count: int,
) -> None:
    """
    Draw the path on the screen using Pygame with a gradient neon glow effect that smoothly transitions through a spectrum of colors. Each segment of the path also changes its color dynamically, creating a gradient of gradients effect. Additionally, implement a fading glow effect for the segments.

    This function meticulously calculates the segment of the path to be displayed, computes the color for each segment based on its position and the frame count, and then draws each segment on the screen. It ensures that the drawing respects the boundaries of the maximum length of the path to be displayed at once and handles the fading effect for older segments.

    Args:
        screen (pygame.Surface): The Pygame screen object where the path will be drawn.
        path (List[Tuple[int, int]]): The path to draw, represented as a list of (x, y) tuples.
        cell_size (int): The size of each cell in the grid, in pixels.
        current_index (int): The current index in the path for the animation, indicating the head of the path.
        max_length (int): The maximum number of segments of the path to display at once.
        frame_count (int): The current frame count, used to adjust the dynamic coloring of the path.

    Returns:
        None: This function does not return any value but directly modifies the screen object passed to it.
    """
    # Determine the segment of the path to be displayed
    start_index = max(0, current_index - max_length)
    end_index = current_index + 1
    visible_segments = path[start_index:end_index]

    # Function to compute the color of a segment based on its index and the frame count
    def compute_color(segment_index: int, frame_count: int) -> Tuple[int, int, int]:
        base_colors = [
            (0, 0, 0),
            (255, 0, 0),
            (255, 165, 0),
            (255, 255, 0),
            (0, 128, 0),
            (0, 0, 255),
            (75, 0, 130),
            (238, 130, 238),
            (255, 255, 255),
            (128, 128, 128),
            (0, 0, 0),
        ]
        num_colors = len(base_colors)
        base_index = segment_index % num_colors
        next_index = (base_index + 1) % num_colors
        ratio = (segment_index % num_colors) / num_colors
        r = int(
            base_colors[base_index][0] * (1 - ratio)
            + base_colors[next_index][0] * ratio
        )
        g = int(
            base_colors[base_index][1] * (1 - ratio)
            + base_colors[next_index][1] * ratio
        )
        b = int(
            base_colors[base_index][2] * (1 - ratio)
            + base_colors[next_index][2] * ratio
        )
        # Adjust color based on frame count to create a dynamic effect
        r = (r + 2 * frame_count) % 256
        g = (g + 2 * frame_count) % 256
        b = (b + 2 * frame_count) % 256
        return (r, g, b)

    # Draw each segment on the screen
    for i, (x, y) in enumerate(visible_segments):
        color = compute_color(i + frame_count, frame_count)
        rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, [min(c + 50, 255) for c in color], rect, 1)
        if i < len(visible_segments) - 1:
            fade_color = [max(c - 10 * (len(visible_segments) - i), 0) for c in color]
            pygame.draw.rect(screen, fade_color, rect, 1)
        logging.debug(f"Segment at {(x, y)} drawn with color {color}.")


def initialize_pygame(
    grid_size: int, cell_size: int
) -> Tuple[pygame.Surface, int, int]:
    """
    Initialize the pygame display environment.

    Args:
    grid_size (int): The size of the grid.
    cell_size (int): The size of each cell in the grid.

    Returns:
    Tuple[pygame.Surface, int, int]: A tuple containing the pygame screen, grid size, and cell size.
    """
    pygame.init()
    screen_dimensions = (grid_size * cell_size, grid_size * cell_size)
    screen = pygame.display.set_mode(screen_dimensions, pygame.RESIZABLE)
    pygame.display.set_caption("Dynamic Hamiltonian Cycle Visualization")
    return screen, grid_size, cell_size


def create_and_convert_grid(grid_size: int) -> Tuple[np.ndarray, nx.Graph]:
    """
    Create a grid and convert it to a graph representation.

    Args:
    grid_size (int): The size of the grid.

    Returns:
    Tuple[np.ndarray, nx.Graph]: A tuple containing the grid as a NumPy array and the graph as a NetworkX graph.
    """
    logging.info("Creating grid and converting to graph using NumPy and NetworkX.")
    grid = create_grid(grid_size)  # Assuming create_grid returns a NumPy array
    graph = convert_to_graph(
        grid
    )  # Assuming convert_to_graph converts a NumPy array to a NetworkX graph
    return grid, graph


def game_loop(
    screen: pygame.Surface,
    current_path: List[Tuple[int, int]],
    cell_size: int,
    path_length: int,
    graph: nx.Graph,
):
    """
    Execute the main game loop.

    Args:
    screen (pygame.Surface): The pygame screen object.
    current_path (List[Tuple[int, int]]): The current path of the Hamiltonian cycle.
    cell_size (int): The size of each cell in the grid.
    path_length (int): The length of the path to display.
    graph (nx.Graph): The graph representing the grid.
    """
    path_index = 0
    running = True
    clock = pygame.time.Clock()
    while running:
        frame_count = pygame.time.get_ticks() // 10
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                logging.info("Quitting the game.")

        screen.fill((0, 0, 0))
        draw_path(screen, current_path, cell_size, path_index, path_length, frame_count)
        pygame.display.flip()
        path_index += 1

        if path_index >= len(current_path) - path_length // 2:
            new_start_vertex = current_path[-1]
            logging.debug(f"Recalculating path from vertex {new_start_vertex}.")
            current_path = current_path[:path_index] + prim_mst(graph, new_start_vertex)

        clock.tick(60)

    pygame.quit()


def main():
    """
    The main function initializes the pygame window, sets up the game grid and graph, and handles the main game loop.
    """
    screen, grid_size, cell_size = initialize_pygame(100, 5)
    grid, graph = create_and_convert_grid(grid_size)
    current_path = prim_mst(
        graph, (0, 0)
    )  # Assuming prim_mst calculates the Minimum Spanning Tree and returns a path
    path_length = 1000
    game_loop(screen, current_path, cell_size, path_length, graph)


if __name__ == "__main__":
    main()
