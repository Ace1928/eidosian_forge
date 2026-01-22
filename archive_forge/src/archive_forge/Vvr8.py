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
        Calculates the Manhattan distance between two nodes.

        Args:
            nodeA (Node): The first node.
            nodeB (Node): The second node.

        Returns:
            int: The Manhattan distance between the two nodes.
        """
        distance_1: int = abs(nodeA.x - nodeB.x)
        distance_2: int = abs(nodeA.y - nodeB.y)
        manhattan_distance: int = distance_1 + distance_2
        logging.debug(
            f"Manhattan distance between {nodeA} and {nodeB} is {manhattan_distance}."
        )
        return manhattan_distance

    def euclidean_distance(self, nodeA: Node, nodeB: Node) -> float:
        """
        Calculates the Euclidean distance between two nodes.

        Args:
            nodeA (Node): The first node.
            nodeB (Node): The second node.

        Returns:
            float: The Euclidean distance between the two nodes.
        """
        distance_1: int = nodeA.x - nodeB.x
        distance_2: int = nodeA.y - nodeB.y
        euclidean_distance: float = math.sqrt(distance_1**2 + distance_2**2)
        logging.debug(
            f"Euclidean distance between {nodeA} and {nodeB} is {euclidean_distance}."
        )
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
        Constructs the path from the given node to the root node.

        Args:
            node (Node): The node from which to construct the path.

        Returns:
            Node: The root node of the path.
        """
        logging.debug("Constructing path from node to root.")
        if node.parent is None:
            logging.debug(f"Node {node} is root node.")
            return node

        while node.parent.parent is not None:
            self.path.append(node)
            logging.debug(f"Added {node} to path.")
            node = node.parent
        logging.debug(f"Path construction completed with final node {node}.")
        return node

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
