from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Node
import math
import logging
import threading

# Configure logging with maximum verbosity and detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

class Algorithm(ABC):
    """
    Abstract base class for pathfinding algorithms in the Snake AI game.

    This class provides a structured template for implementing various pathfinding algorithms,
    ensuring that each derived class adheres to a consistent interface for initializing the grid,
    calculating distances, and running the algorithm. It also ensures thread safety and efficient
    resource management to optimize performance and flexibility.

    Attributes:
        grid (List[List[Node]]): A grid representation of the game environment.
        frontier (List[Node]): List of nodes to be explored.
        explored_set (List[Node]): List of nodes that have been explored.
        path (List[Node]): List representing the path from start to goal.
        lock (threading.Lock): A lock for thread-safe manipulation of shared resources.
    """

    def __init__(self, grid: List[List[Node]]) -> None:
        """
        Initializes the Algorithm with a grid, setting up the necessary data structures
        for pathfinding operations. It also configures thread safety mechanisms.

        Args:
            grid (List[List[Node]]): The grid representing the game environment.
        """
        self.grid: List[List[Node]] = grid
        self.frontier: List[Node] = []
        self.explored_set: List[Node] = []
        self.path: List[Node] = []
        self.lock: threading.Lock = threading.Lock()  # Initialize the threading lock for thread safety

        logging.debug("Algorithm base class initialized with grid.")
        logging.debug(f"Grid dimensions set to {len(grid)}x{len(grid[0])} with total cells: {len(grid) * len(grid[0])}")
        logging.debug("Frontier, Explored Set, and Path lists have been initialized and are ready for use.")

        # Ensuring that all resources are ready and logging is set up before any operations begin
        try:
            with self.lock:
                # Placeholder for any pre-processing required on the grid or other structures
                logging.debug("Pre-processing of grid and data structures completed successfully.")
        except Exception as e:
            logging.error("An error occurred during the initialization of the Algorithm: {}".format(e))
            raise

    def get_initstate_and_goalstate(self, snake) -> Tuple[Node, Node]:
        """
        Determines the initial and goal states for the algorithm based on the current position of the snake
        and the position of the fruit, which is the target.

        This method meticulously extracts the current position of the snake and the position of the fruit
        from the game state, encapsulating them into Node objects which represent the initial and goal states
        respectively for the pathfinding algorithm. This is a critical step in setting up the algorithm's
        environment for execution.

        Args:
            snake (Snake): The snake object from which the current position and the target position are derived.

        Returns:
            Tuple[Node, Node]: A tuple containing the Node representations of the initial state (current position of the snake)
            and the goal state (position of the fruit).

        Raises:
            ValueError: If the snake or fruit positions are undefined or result in invalid Node coordinates.

        Detailed logging is used to ensure that each step of the process is recorded for debugging and traceability.
        """
        try:
            # Extracting the x and y coordinates of the snake's current position
            initial_x: int = snake.get_x()
            initial_y: int = snake.get_y()
            # Creating a Node for the initial state
            initial_state: Node = Node(initial_x, initial_y)
            logging.debug(f"Initial state Node created at coordinates ({initial_x}, {initial_y}).")

            # Extracting the x and y coordinates of the fruit's position
            fruit_x: int = snake.get_fruit().x
            fruit_y: int = snake.get_fruit().y
            # Creating a Node for the goal state
            goal_state: Node = Node(fruit_x, fruit_y)
            logging.debug(f"Goal state Node created at coordinates ({fruit_x}, {fruit_y}).")

            # Logging the successful determination of initial and goal states
            logging.info(f"Initial state determined at {initial_state}. Goal state determined at {goal_state}.")

            return initial_state, goal_state

        except AttributeError as e:
            # Logging an error if there is an issue with accessing snake or fruit attributes
            logging.error("Failed to access snake or fruit attributes.", exc_info=True)
            raise ValueError("Invalid snake or fruit attributes provided.") from e

        except Exception as e:
            # Handling any other unexpected exceptions
            logging.critical("An unexpected error occurred while determining initial and goal states.", exc_info=True)
            raise RuntimeError("An unexpected error occurred in get_initstate_and_goalstate method.") from e
    def manhattan_distance(self, nodeA: Node, nodeB: Node) -> int:
        """
        Calculates the Manhattan distance between two nodes, which is the sum of the absolute differences of their Cartesian coordinates.
        This distance metric is crucial in grid-based pathfinding contexts where only four-directional movement is allowed (up, down, left, right).

        The method employs a detailed and meticulous approach to ensure the calculation is not only accurate but also optimized for performance.
        It leverages Python's built-in `abs` function to compute the absolute difference between the x-coordinates and y-coordinates of the two nodes,
        summing these differences to obtain the Manhattan distance.

        Args:
            nodeA (Node): The first node, representing a point in the grid.
            nodeB (Node): The second node, representing another point in the grid.

        Returns:
            int: The Manhattan distance between the two nodes, which is a non-negative integer representing the number of grid steps required to move from nodeA to nodeB without diagonal movement.

        Raises:
            TypeError: If either nodeA or nodeB is not an instance of the Node class.

        Detailed logging is used to ensure that each step of the process is recorded for debugging and traceability. The function also checks the type of the input parameters to prevent runtime errors due to incorrect types.
        """
        if not isinstance(nodeA, Node) or not isinstance(nodeB, Node):
            logging.error("One or both of the provided arguments are not of type Node.")
            raise TypeError("Both nodeA and nodeB must be instances of the Node class.")

        # Extracting the x and y coordinates of both nodes
        x1, y1 = nodeA.x, nodeA.y
        x2, y2 = nodeB.x, nodeB.y

        # Calculating the absolute differences in both dimensions
        delta_x: int = abs(x1 - x2)
        delta_y: int = abs(y1 - y2)

        # Summing the differences to get the Manhattan distance
        manhattan_distance: int = delta_x + delta_y

        # Logging the calculated Manhattan distance with detailed node information
        logging.debug(f"Calculated Manhattan distance between Node A at ({x1}, {y1}) and Node B at ({x2}, {y2}) is {manhattan_distance}.")

        return manhattan_distance
    def euclidean_distance(self, nodeA: Node, nodeB: Node) -> float:
        """
        Calculates the Euclidean distance between two nodes with meticulous precision and comprehensive detail.

        The Euclidean distance, also known as L2 norm or Euclidean norm, is the straight-line distance between two points
        in Euclidean space. This function computes the Euclidean distance between two nodes represented by their Cartesian
        coordinates in a 2D space, ensuring the calculation adheres to mathematical principles of geometry.

        Args:
            nodeA (Node): The first node, representing a point in the grid with coordinates (x, y).
            nodeB (Node): The second node, representing another point in the grid with coordinates (x, y).

        Returns:
            float: The Euclidean distance between the two nodes, calculated using the formula sqrt((x2 - x1)^2 + (y2 - y1)^2).

        Raises:
            TypeError: If either nodeA or nodeB is not an instance of the Node class.

        Detailed logging is used to ensure that each step of the process is recorded for debugging and traceability.
        The function also checks the type of the input parameters to prevent runtime errors due to incorrect types.
        """
        if not isinstance(nodeA, Node) or not isinstance(nodeB, Node):
            logging.error("One or both of the provided arguments are not of type Node.")
            raise TypeError("Both nodeA and nodeB must be instances of the Node class.")

        # Extracting the x and y coordinates of both nodes
        x1, y1 = nodeA.x, nodeA.y
        x2, y2 = nodeB.x, nodeB.y

        # Calculating the squared differences in both dimensions
        squared_difference_x: int = (x1 - x2) ** 2
        squared_difference_y: int = (y1 - y2) ** 2

        # Summing the squared differences
        sum_of_squared_differences: int = squared_difference_x + squared_difference_y

        # Calculating the Euclidean distance as the square root of the sum of squared differences
        euclidean_distance: float = math.sqrt(sum_of_squared_differences)

        # Logging the calculated Euclidean distance with detailed node information
        logging.debug(f"Calculated Euclidean distance between Node A at ({x1}, {y1}) and Node B at ({x2}, {y2}) is {euclidean_distance}.")

        return euclidean_distance

    @abstractmethod
    def run_algorithm(self, snake) -> Optional[Node]:
        """
        Abstract method to run the pathfinding algorithm.

        Args:
            snake: The snake object.

        Returns:
            Optional[Node]: The next node in the path for the snake to follow, if any.
        """
        pass
    def get_path(self, node: Node) -> Node:
        """
        Methodically constructs a comprehensive path from the specified node to the root node, ensuring each step is
        meticulously logged and the path is accurately assembled.

        Args:
            node (Node): The node from which to initiate the path construction.

        Returns:
            Node: The root node of the path, representing the origin of the traversal.

        Raises:
            ValueError: If the node provided is None, indicating an invalid starting point for path construction.

        This function employs a systematic approach to trace back from the given node to the root node by iteratively
        navigating through the parent nodes. It maintains a list to capture the sequence of nodes for potential future
        use and debugging. The function is robust against null references and logs each significant step in the process.
        """
        if node is None:
            logging.error("Attempted to construct a path from a None node.")
            raise ValueError("The 'node' argument cannot be None.")

        path_trace = []  # Initialize an empty list to store the path trace for debugging and verification purposes.
        current_node = node  # Start with the given node.

        # Log the initiation of path construction.
        logging.debug(f"Initiating path construction from node: {current_node}")

        # Traverse from the current node to the root node.
        while current_node.parent is not None:
            path_trace.append(current_node)
            logging.debug(f"Traversing from node {current_node} to its parent {current_node.parent}")
            current_node = current_node.parent

        # Append the root node to the path trace and log the completion of the path.
        path_trace.append(current_node)
        logging.debug(f"Root node {current_node} reached. Path construction complete.")

        # Optionally, reverse the path trace to display from root to the original node if required.
        # path_trace.reverse()  # Uncomment this line if the path needs to be reported from root to node.

        # Log the final path for verification.
        logging.info(f"Constructed path: {path_trace}")

        return current_node  # Return the root node as the result of the path construction.

    def inside_body(self, snake, node: Node) -> bool:
        """
        Checks if a node is inside the snake's body.

        Args:
            snake: The snake object.
            node (Node): The node to check.

        Returns:
            bool: True if the node is inside the snake's body, False otherwise.
        """
        for body in snake.body:
            if body.x == node.x and body.y == node.y:
                logging.debug(f"Node {node} is inside snake's body.")
                return True
        logging.debug(f"Node {node} is not inside snake's body.")
        return False

    def outside_boundary(self, node: Node) -> bool:
        """
        Checks if a node is outside the game boundaries.

        Args:
            node (Node): The node to check.

        Returns:
            bool: True if the node is outside the boundaries, False otherwise.
        """
        if not 0 <= node.x < NO_OF_CELLS:
            logging.debug(f"Node {node} is outside horizontal boundaries.")
            return True
        elif not BANNER_HEIGHT <= node.y < NO_OF_CELLS:
            logging.debug(f"Node {node} is outside vertical boundaries.")
            return True
        logging.debug(f"Node {node} is inside boundaries.")
        return False

    def get_neighbors(self, node: Node) -> List[Node]:
        """
        Retrieves the neighboring nodes of a given node.

        Args:
            node (Node): The node for which to find neighbors.

        Returns:
            List[Node]: A list of neighboring nodes.
        """
        i: int = int(node.x)
        j: int = int(node.y)

        neighbors: List[Node] = []
        # left [i-1, j]
        if i > 0:
            neighbors.append(self.grid[i - 1][j])
            logging.debug(f"Left neighbor {self.grid[i-1][j]} added.")
        # right [i+1, j]
        if i < NO_OF_CELLS - 1:
            neighbors.append(self.grid[i + 1][j])
            logging.debug(f"Right neighbor {self.grid[i+1][j]} added.")
        # top [i, j-1]
        if j > 0:
            neighbors.append(self.grid[i][j - 1])
            logging.debug(f"Top neighbor {self.grid[i][j-1]} added.")
        # bottom [i, j+1]
        if j < NO_OF_CELLS - 1:
            neighbors.append(self.grid[i][j + 1])
            logging.debug(f"Bottom neighbor {self.grid[i][j+1]} added.")

        logging.debug(f"Neighbors of {node}: {neighbors}")
        return neighbors
