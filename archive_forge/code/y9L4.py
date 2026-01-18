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
from functools import (
    lru_cache,
)  # Importing lru_cache for caching function outputs to optimize performance.

# Initialize logging with maximum verbosity and detail
logging.basicConfig(
    level=logging.DEBUG,  # Setting the logging level to DEBUG to capture all levels of log messages.
    format="%(asctime)s - %(levelname)s - %(message)s",  # Defining the format of log messages to include the time, the level of severity, and the message.
)


@lru_cache(maxsize=None)
def create_grid(size: int) -> np.ndarray:
    """
    Constructs a two-dimensional square grid of a specified size where each cell within the grid is initialized to zero. This grid is represented as a NumPy array, which is chosen for its optimized operations and superior performance characteristics, especially beneficial for handling large grids efficiently.

    Args:
        size (int): The dimension of the grid, which is used to define both the number of rows and the columns, given the grid is square in shape.

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


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
def prim_mst(graph: nx.Graph, start_vertex: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Calculate the Minimum Spanning Tree (MST) using Prim's algorithm starting from a given vertex.

    Args:
        graph (nx.Graph): The graph on which to perform the MST calculation.
        start_vertex (Tuple[int, int]): The starting vertex for the MST.

    Returns:
        List[Tuple[int, int]]: The path representing the MST as a list of vertices.
    """
    mst = []
    visited = set([start_vertex])
    edges = [(weight, start_vertex, to) for to, weight in graph[start_vertex].items()]
    heapq.heapify(edges)

    while edges:
        weight, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.append(to)
            for next_to, weight in graph[to].items():
                if next_to not in visited:
                    heapq.heappush(edges, (weight, to, next_to))

    return mst


@lru_cache(maxsize=None)
def hamiltonian_cycle(graph: nx.Graph) -> List[Tuple[int, int]]:
    """
    Find a Hamiltonian cycle in a graph using the Nearest Neighbor heuristic.

    Args:
        graph (nx.Graph): The graph in which to find the Hamiltonian cycle.

    Returns:
        List[Tuple[int, int]]: The Hamiltonian cycle as a list of vertices.
    """
    cycle = []
    visited = set()
    current_vertex = random.choice(list(graph.nodes))
    visited.add(current_vertex)
    cycle.append(current_vertex)

    while len(visited) < len(graph.nodes):
        neighbors = list(graph[current_vertex].keys())
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        if unvisited_neighbors:
            next_vertex = min(
                unvisited_neighbors, key=lambda n: graph[current_vertex][n]["weight"]
            )
        else:
            next_vertex = min(
                neighbors, key=lambda n: graph[current_vertex][n]["weight"]
            )
        cycle.append(next_vertex)
        visited.add(next_vertex)
        current_vertex = next_vertex

    cycle.append(cycle[0])
    return cycle


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
        """
        Compute the RGB color for a segment based on its index and the frame count, utilizing a gradient transition between predefined base colors. This function ensures that the color transitions smoothly and dynamically adjusts based on the frame count to create a visually appealing effect.

        Args:
            segment_index (int): The index of the segment for which the color needs to be computed.
            frame_count (int): The current frame count used to dynamically adjust the color.

        Returns:
            Tuple[int, int, int]: A tuple representing the RGB color calculated for the given segment index and frame count.

        Detailed Description:
            - The function defines a list of base colors which are used to create a gradient effect.
            - It calculates the indices for the current and next color in the base colors list based on the segment index.
            - A ratio is computed to determine the blend between the current and next color.
            - RGB values are interpolated based on this ratio and then adjusted dynamically based on the frame count to ensure the color changes over time, enhancing the visual dynamics of the display.
            - The function returns the dynamically computed RGB color as a tuple.
        """
        # Define the base colors for the gradient transition
        base_colors = np.array(
            [
                [0, 0, 0],
                [255, 0, 0],
                [255, 165, 0],
                [255, 255, 0],
                [0, 128, 0],
                [0, 0, 255],
                [75, 0, 130],
                [238, 130, 238],
                [255, 255, 255],
                [128, 128, 128],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )

        # Calculate indices for interpolation
        num_colors = base_colors.shape[0]
        base_index = segment_index % num_colors
        next_index = (base_index + 1) % num_colors

        # Compute interpolation ratio
        ratio = (segment_index % num_colors) / float(num_colors)

        # Interpolate RGB values
        r = int(
            base_colors[base_index, 0] * (1 - ratio)
            + base_colors[next_index, 0] * ratio
        )
        g = int(
            base_colors[base_index, 1] * (1 - ratio)
            + base_colors[next_index, 1] * ratio
        )
        b = int(
            base_colors[base_index, 2] * (1 - ratio)
            + base_colors[next_index, 2] * ratio
        )

        # Dynamic color adjustment based on frame count
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
    Methodically initializes the pygame display environment with a specific grid size and cell size, configuring the display settings to accommodate a visualization of a dynamic Hamiltonian cycle. This function meticulously sets up the graphical interface, ensuring that the display is both responsive and appropriately scaled according to the provided dimensions.

    Args:
        grid_size (int): The size of the grid, which determines the number of cells in both the horizontal and vertical dimensions of the grid.
        cell_size (int): The size of each individual cell within the grid, dictating the pixel dimensions of each cell.

    Returns:
        Tuple[pygame.Surface, int, int]: A tuple containing the initialized pygame screen object, the grid size, and the cell size. This tuple provides the necessary components to interact with the pygame environment effectively.

    Detailed Description:
        - The function begins by invoking `pygame.init()` to initialize all imported pygame modules in a safe manner, preparing the system for further configuration and use.
        - It then calculates the dimensions of the screen based on the grid size and cell size. This calculation is performed by multiplying the grid size by the cell size for both width and height, resulting in a tuple that specifies the full pixel dimensions of the display.
        - A pygame display mode is then set with these dimensions, and the display is configured to be resizable, allowing for dynamic adjustment of the window size by the user.
        - The display's caption is set to "Dynamic Hamiltonian Cycle Visualization" to provide context to the user regarding the content being visualized.
        - The function concludes by returning a tuple containing the screen object, grid size, and cell size, encapsulating all necessary information for managing the display in subsequent operations.

    Raises:
        Exception: If an error occurs during the initialization of the pygame modules or the configuration of the display settings, an exception will be raised to indicate the failure of the setup process.

    Logging:
        - The function logs detailed debug information at each step to provide insights into the execution flow and to assist in troubleshooting potential issues in the setup process.
    """
    logging.debug("Initializing pygame modules.")
    pygame.init()
    logging.debug(f"Pygame initialized successfully.")

    screen_dimensions = (grid_size * cell_size, grid_size * cell_size)
    logging.debug(f"Screen dimensions calculated as: {screen_dimensions}")

    try:
        screen = pygame.display.set_mode(screen_dimensions, pygame.RESIZABLE)
        logging.debug("Display mode set successfully.")
    except Exception as e:
        logging.error(f"Failed to set display mode: {str(e)}")
        raise

    pygame.display.set_caption("Dynamic Hamiltonian Cycle Visualization")
    logging.debug("Display caption set to 'Dynamic Hamiltonian Cycle Visualization'.")

    logging.info("Pygame display environment initialized successfully.")
    return screen, grid_size, cell_size


def create_and_convert_grid(grid_size: int) -> Tuple[np.ndarray, nx.Graph]:
    """
    Methodically constructs a two-dimensional grid based on the specified size and subsequently transforms this grid into a graph representation utilizing the NetworkX library. This function is designed to operate with high efficiency and precision, leveraging the capabilities of NumPy for array manipulations and NetworkX for graph operations, ensuring that the transformation is both accurate and optimal.

    Args:
        grid_size (int): The dimension of the grid which specifies both the width and height as the grid is square.

    Returns:
        Tuple[np.ndarray, nx.Graph]: A tuple where the first element is a NumPy array representing the grid and the second element is a NetworkX graph derived from the grid structure.

    Raises:
        ValueError: If the grid_size is less than 1, as a grid with non-positive dimensions is not permissible.

    Detailed Description:
        - The function initiates by logging the commencement of the grid creation and graph conversion process.
        - It calls the `create_grid` function, which is expected to return a NumPy array representing a grid initialized to zero. This grid serves as the foundational data structure for subsequent operations.
        - Following the grid creation, the `convert_to_graph` function is invoked, which takes the NumPy array as input and returns a NetworkX graph. This graph encapsulates the connectivity and structure of the grid in a format suitable for advanced graph-theoretical operations.
        - Throughout the process, detailed debug logging captures key steps and data states to facilitate troubleshooting and verification of operations.
        - The function concludes by returning a tuple containing the grid and the graph, ensuring that both data structures are readily accessible for further processing.
    """
    logging.info(
        f"Initiating the creation of a grid and its conversion to a graph with a grid size of {grid_size}."
    )

    try:
        grid = create_grid(grid_size)
        logging.debug(f"Grid created successfully with size {grid_size}x{grid_size}.")
    except ValueError as e:
        logging.error(
            f"Failed to create grid due to invalid size: {grid_size}. Error: {str(e)}"
        )
        raise

    try:
        graph = convert_to_graph(grid)
        logging.debug("Conversion of grid to graph completed successfully.")
    except Exception as e:
        logging.error("Failed to convert grid to graph. Error: " + str(e))
        raise

    logging.info("Grid and graph have been successfully created and converted.")
    return grid, graph


def game_loop(
    screen: pygame.Surface,
    current_path: np.ndarray,
    cell_size: int,
    path_length: int,
    graph: nx.Graph,
):
    """
    Execute the main game loop, which is responsible for continuously updating the display of the Hamiltonian cycle path on the screen until the termination condition is met.

    Args:
        screen (pygame.Surface): The pygame screen object which acts as the canvas for drawing the game elements.
        current_path (np.ndarray): The current path of the Hamiltonian cycle represented as a NumPy array of tuples, where each tuple contains the x and y coordinates of a vertex in the path.
        cell_size (int): The size of each cell in the grid, which determines the scale of the drawing on the screen.
        path_length (int): The length of the path to display, which controls how much of the path is visible at any given time.
        graph (nx.Graph): The graph representing the grid, used for recalculating paths as needed.

    Detailed Description:
        - The function initializes the game state and enters a loop that continues until the pygame.QUIT event is triggered.
        - Within the loop, the screen is cleared and the current segment of the path is drawn using the draw_path function.
        - The display is updated with pygame.display.flip(), and the path_index is incremented to advance the path visualization.
        - If the path_index reaches a certain threshold, the path is recalculated from the current vertex using the prim_mst function.
        - The loop runs at a controlled frame rate of 60 FPS using pygame.time.Clock().
        - Extensive logging is used to record the state of the game, including path recalculations and game termination.
    """
    path_index = 0
    running = True
    clock = pygame.time.Clock()
    while running:
        frame_count = pygame.time.get_ticks() // 10
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                logging.info("Quitting the game loop due to user request.")

        screen.fill((0, 0, 0))  # Fill the screen with black before drawing
        draw_path(screen, current_path, cell_size, path_index, path_length, frame_count)
        pygame.display.flip()  # Update the full display Surface to the screen
        path_index = (path_index + 1) % len(
            current_path
        )  # Ensure the path_index wraps around

        if path_index >= len(current_path) - path_length // 2:
            new_start_vertex = current_path[path_index % len(current_path)]
            logging.debug(
                f"Recalculating path from vertex {new_start_vertex} due to reaching path display threshold."
            )
            current_path = np.concatenate(
                (current_path[:path_index], prim_mst(graph, new_start_vertex))
            )

        clock.tick(60)  # Maintain a consistent frame rate of 60 FPS

    pygame.quit()  # Uninitialize all pygame modules


def main():
    """
    The main function initializes the pygame window, sets up the game grid and graph, and handles the main game loop.

    Detailed Description:
        - The function begins by initializing the pygame window with specific dimensions and cell size by invoking the `initialize_pygame` function.
        - It then proceeds to create a grid and convert this grid into a graph representation suitable for computational operations, specifically for pathfinding algorithms.
        - Utilizing the `prim_mst` function, it calculates the Minimum Spanning Tree (MST) starting from the origin of the grid (0, 0) and derives a path from this MST.
        - This path is then utilized in the `game_loop` function, which continuously updates and displays the path on the pygame window.
        - The function ensures that all operations are logged with high verbosity for detailed tracing and debugging.
        - Exception handling is robust, ensuring that any errors during the game initialization or execution are caught and logged, and the system exits gracefully.

    Args:
        None

    Returns:
        None

    Raises:
        Exception: Catches and logs any exceptions that occur during the initialization or execution of the game loop, ensuring the program does not crash unexpectedly and that all resources are properly cleaned up.
    """
    try:
        logging.info(
            "Initializing the pygame window with specified dimensions and cell size."
        )
        screen, grid_size, cell_size = initialize_pygame(100, 5)
        logging.info("Creating game grid and converting it to a graph representation.")
        grid, graph = create_and_convert_grid(grid_size)
        logging.info(
            "Calculating the Minimum Spanning Tree to determine the initial path."
        )
        current_path = prim_mst(graph, (0, 0))
        path_length = 1000
        logging.info(
            "Entering the main game loop to display the path and handle user interactions."
        )
        game_loop(screen, current_path, cell_size, path_length, graph)
    except Exception as e:
        logging.error(f"An error occurred during game initialization or execution: {e}")
        raise


if __name__ == "__main__":
    logging.info("Starting the Hamiltonian Screen Saver application.")
    main()
    main()
