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
        Initializes the Algorithm with a grid, meticulously setting up the necessary data structures
        for pathfinding operations. It also configures thread safety mechanisms with precision and ensures
        that all components are ready for operation with comprehensive logging for traceability.

        Args:
            grid (List[List[Node]]): The grid representing the game environment, which is a two-dimensional
            list of Node objects that encapsulate the state of each cell in the game environment.

        Raises:
            Exception: Propagates exceptions that may occur during the initialization process, providing
            detailed error logging for diagnostic purposes.
        """
        # Initialize the grid and associated data structures with explicit type annotations
        self.grid: List[List[Node]] = grid
        self.frontier: List[Node] = []
        self.explored_set: List[Node] = []
        self.path: List[Node] = []
        self.lock: threading.Lock = threading.Lock()  # Initialize the threading lock for thread safety

        # Log the initialization details with high verbosity
        logging.debug("Algorithm base class initialized with grid.")
        logging.debug(f"Grid dimensions set to {len(grid)}x{len(grid[0])} with total cells: {len(grid) * len(grid[0])}")
        logging.debug("Frontier, Explored Set, and Path lists have been initialized and are ready for use.")

    def get_initstate_and_goalstate(self, snake) -> Tuple[Node, Node]:
        """
        Determines the initial and goal states for the algorithm by extracting the current position of the snake
        and the position of the fruit, which represents the target destination.

        Args:
            snake: The snake object, which encapsulates the current state and position of the snake in the game.

        Returns:
            Tuple[Node, Node]: A tuple containing the initial and goal state nodes, which are critical for guiding
            the pathfinding algorithm in computing the optimal path from the snake's current position to the fruit.

        Raises:
            ValueError: If the snake or fruit positions are undefined or result in invalid node coordinates, an error
            is raised to prevent erroneous pathfinding operations.
        """
        try:
            initial_state: Node = Node(snake.get_x(), snake.get_y())
            goal_state: Node = Node(snake.get_fruit().x, snake.get_fruit().y)
            logging.debug(f"Initial state determined at {initial_state}. Goal state determined at {goal_state}.")
            return initial_state, goal_state
        except AttributeError as e:
            logging.error(f"Failed to retrieve initial or goal state due to: {e}", exc_info=True)
            raise ValueError(f"Invalid snake or fruit position: {e}") from e
    
    def manhattan_distance(self, nodeA: Node, nodeB: Node) -> int:
        """
        Calculates the Manhattan distance between two nodes, which is the sum of the absolute differences
        of their Cartesian coordinates. This distance metric is crucial in grid-based pathfinding algorithms
        where only four-directional movement is allowed.

        Args:
            nodeA (Node): The first node, representing a specific coordinate in the grid.
            nodeB (Node): The second node, representing another coordinate in the grid.

        Returns:
            int: The Manhattan distance between the two nodes, which is a non-negative integer representing
            the minimum number of moves required to travel between the two points in a grid if only four-directional
            movements are allowed.

        Raises:
            TypeError: If either nodeA or nodeB is not an instance of Node, indicating improper usage of the function.

        Detailed Description:
            The Manhattan distance is particularly useful in scenarios where movement is restricted to horizontal
            and vertical directions only, such as in many board games, pixel grids, and urban layouts. This function
            ensures that the nodes provided are of the correct type and computes the distance in a robust manner,
            logging detailed debug information to facilitate troubleshooting and verification of operations.
        """
        if not isinstance(nodeA, Node) or not isinstance(nodeB, Node):
            logging.error("Manhattan distance calculation received non-Node inputs.")
            raise TypeError("Both nodeA and nodeB must be instances of Node for Manhattan distance calculation.")
        
        # Calculate the absolute differences in the x and y coordinates
        distance_x: int = abs(nodeA.x - nodeB.x)
        distance_y: int = abs(nodeA.y - nodeB.y)
        
        # Sum the absolute differences to get the Manhattan distance
        calculated_manhattan_distance: int = distance_x + distance_y
        
        # Log the detailed debug information
        logging.debug(f"Calculated Manhattan distance from Node({nodeA.x}, {nodeA.y}) to Node({nodeB.x}, {nodeB.y}) is {calculated_manhattan_distance}.")
        
        # Return the calculated Manhattan distance
        return calculated_manhattan_distance
    
    def euclidean_distance(self, nodeA: Node, nodeB: Node) -> float:
        """
        Calculates the Euclidean distance between two nodes with detailed logging and error handling.

        The Euclidean distance is the "ordinary" straight-line distance between two points in Euclidean space.
        This function computes the Euclidean distance between two nodes, each represented by Cartesian coordinates
        (x, y), in a robust and error-resistant manner, ensuring that the nodes are valid and that the computation
        adheres to the highest standards of precision and accuracy.

        Args:
            nodeA (Node): The first node, representing a specific coordinate in the grid.
            nodeB (Node): The second node, representing another coordinate in the grid.

        Returns:
            float: The Euclidean distance between the two nodes, computed as the square root of the sum of the
            squared differences in their Cartesian coordinates.

        Raises:
            TypeError: If either nodeA or nodeB is not an instance of Node, indicating improper usage of the function.

        Detailed Description:
            This function first validates the input to ensure that both nodeA and nodeB are instances of the Node class.
            It then calculates the differences in the x and y coordinates of the nodes, squares these differences,
            and sums them. The square root of this sum is then computed to yield the Euclidean distance, which is
            returned as a floating-point number. Throughout this process, detailed debug information is logged,
            and appropriate exceptions are raised and logged in case of errors.

        Example:
            node1 = Node(1, 2)
            node2 = Node(4, 6)
            distance = euclidean_distance(node1, node2)
            print(distance)  # Output: 5.0
        """
        if not isinstance(nodeA, Node) or not isinstance(nodeB, Node):
            logging.error("Euclidean distance calculation received non-Node inputs.")
            raise TypeError("Both nodeA and nodeB must be instances of Node for Euclidean distance calculation.")

        try:
            # Calculate the differences in the x and y coordinates
            difference_x: int = nodeA.x - nodeB.x
            difference_y: int = nodeA.y - nodeB.y

            # Compute the squares of the differences
            squared_difference_x: int = difference_x ** 2
            squared_difference_y: int = difference_y ** 2

            # Sum the squared differences
            sum_of_squared_differences: int = squared_difference_x + squared_difference_y

            # Calculate the square root of the sum of squared differences to get the Euclidean distance
            calculated_euclidean_distance: float = math.sqrt(sum_of_squared_differences)

            # Log the detailed debug information
            logging.debug(f"Calculated Euclidean distance from Node({nodeA.x}, {nodeA.y}) to Node({nodeB.x}, {nodeB.y}) is {calculated_euclidean_distance}.")

            # Return the calculated Euclidean distance
            return calculated_euclidean_distance
        except Exception as e:
            logging.error(f"An error occurred while calculating Euclidean distance: {e}", exc_info=True)
            raise
   
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
        Constructs the path from the specified node to the root node, meticulously tracing back through the parent nodes.

        This method initiates at the given node and traverses backwards through its parent nodes until it reaches the root node,
        which is identified by the absence of a parent. Each node encountered is appended to the path list maintained by the
        Algorithm class. This method is thread-safe, ensuring that the path list is manipulated without concurrency issues.

        Args:
            node (Node): The node from which to start constructing the path.

        Returns:
            Node: The root node of the path, which is the node in the path without a parent.

        Raises:
            ValueError: If the input node is None, indicating an invalid starting point for path construction.
        """
        if node is None:
            logging.error("Attempted to get path starting from a None node.")
            raise ValueError("The input node cannot be None.")

        logging.debug(f"Starting path construction from node ({node.x}, {node.y}).")
        
        # Thread-safe manipulation of shared path list
        with self.lock:
            # Clear the path list to ensure fresh computation of the path
            self.path.clear()
            logging.debug("Cleared existing path data for fresh path computation.")

            # Traverse from the given node up to the root node
            current_node: Node = node
            while current_node.parent is not None:
                self.path.append(current_node)
                logging.debug(f"Added node ({current_node.x}, {current_node.y}) to path.")
                current_node = current_node.parent

            # Append the root node to the path
            self.path.append(current_node)
            logging.debug(f"Root node ({current_node.x}, {current_node.y}) added to path.")

        logging.debug("Path construction completed. Path details:")
        for node in self.path:
            logging.debug(f"Node in path: ({node.x}, {node.y}) with parent ({node.parent.x if node.parent else 'None'}, {node.parent.y if node.parent else 'None'}).")

        # Return the root node, which is the last node added to the path
        return self.path[-1]

    def inside_body(self, snake: 'Snake', node: Node) -> bool:
        """
        Determines with meticulous precision whether a given node is located within the snake's body in the game grid.

        This method iterates over each segment of the snake's body and compares the coordinates of the segment with
        those of the provided node. If a match is found, indicating that the node is indeed part of the snake's body,
        a detailed debug log is recorded, and the method returns True. If no match is found after checking all segments,
        it logs this outcome and returns False.

        Args:
            snake (Snake): The snake object, representing the snake in the game, which consists of multiple body segments.
            node (Node): The node to check, representing a specific position in the game grid.

        Returns:
            bool: True if the node is inside the snake's body, False otherwise.

        Raises:
            TypeError: If the 'node' is not an instance of Node or 'snake' does not have a 'body' attribute.

        Detailed logging is used to trace the method's execution path and outcomes for diagnostic and debugging purposes.
        """
        if not isinstance(node, Node):
            logging.error(f"Expected 'node' to be an instance of Node, got {type(node)} instead.")
            raise TypeError(f"Expected 'node' to be an instance of Node, got {type(node)} instead.")

        if not hasattr(snake, 'body') or not isinstance(snake.body, list):
            logging.error("The snake object does not have a 'body' attribute or it is not of type list.")
            raise AttributeError("The snake object must have a 'body' attribute of type list.")

        for segment in snake.body:
            if segment.x == node.x and segment.y == node.y:
                logging.debug(f"Node ({node.x}, {node.y}) is inside the snake's body at segment ({segment.x}, {segment.y}).")
                return True
            else:
                logging.debug(f"Node ({node.x}, {node.y}) checked against segment ({segment.x}, {segment.y}) - no match.")

        logging.debug(f"Node ({node.x}, {node.y}) is not inside the snake's body after checking all segments.")
        return False
    def outside_boundary(self, node: Node) -> bool:
        """
        Determines with utmost precision whether a given node is positioned outside the predefined game boundaries,
        which are defined by the constants NO_OF_CELLS and BANNER_HEIGHT.

        This method meticulously checks the node's coordinates against the horizontal and vertical boundaries of the
        game grid. It employs detailed logging to trace the node's position relative to these boundaries, ensuring
        comprehensive data capture and diagnostic capabilities.

        Args:
            node (Node): The node whose position is to be verified against the game's boundary conditions.

        Returns:
            bool: True if the node is outside the defined boundaries, False otherwise.

        Raises:
            ValueError: If the 'node' is not an instance of Node, ensuring type safety.

        Detailed logging is used to trace the method's execution path and outcomes for diagnostic and debugging purposes.
        """
        if not isinstance(node, Node):
            logging.error(f"Expected 'node' to be an instance of Node, received {type(node)} instead.")
            raise ValueError(f"Expected 'node' to be an instance of Node, received {type(node)} instead.")

        # Check horizontal boundaries
        if not 0 <= node.x < NO_OF_CELLS:
            logging.debug(f"Node ({node.x}, {node.y}) is outside horizontal boundaries (0, {NO_OF_CELLS-1}).")
            return True
        # Check vertical boundaries
        elif not BANNER_HEIGHT <= node.y < NO_OF_CELLS:
            logging.debug(f"Node ({node.x}, {node.y}) is outside vertical boundaries ({BANNER_HEIGHT}, {NO_OF_CELLS-1}).")
            return True

        # If the node is within all boundaries
        logging.debug(f"Node ({node.x}, {node.y}) is inside the game boundaries.")
        return False
    def get_neighbors(self, node: Node) -> List[Node]:
        """
        Retrieves the neighboring nodes of a given node with meticulous precision and comprehensive detail.

        This method calculates the adjacent nodes (left, right, top, bottom) of the specified node within the grid,
        ensuring that the boundaries are respected. It employs rigorous checks to prevent accessing nodes outside the
        grid boundaries, thereby maintaining the integrity and correctness of the algorithm's spatial logic.

        Args:
            node (Node): The node for which to find neighbors, must be a valid node within the grid.

        Returns:
            List[Node]: A list containing the neighboring nodes that are within the valid range of the grid.

        Raises:
            ValueError: If the provided node is not within the valid range of the grid.

        Detailed logging is used to trace the method's execution path and outcomes for diagnostic and debugging purposes.
        """
        if not isinstance(node, Node):
            logging.error(f"Invalid type for node: expected Node, got {type(node).__name__}")
            raise ValueError(f"Invalid type for node: expected Node, got {type(node).__name__}")

        i: int = int(node.x)
        j: int = int(node.y)

        if not (0 <= i < NO_OF_CELLS) or not (0 <= j < NO_OF_CELLS):
            logging.error(f"Node coordinates ({i}, {j}) are out of the grid boundaries.")
            raise ValueError(f"Node coordinates ({i}, {j}) are out of the grid boundaries.")

        neighbors: List[Node] = []
        directions: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        direction_names: List[str] = ["Left", "Right", "Top", "Bottom"]

        for offset, name in zip(directions, direction_names):
            ni, nj = i + offset[0], j + offset[1]
            if 0 <= ni < NO_OF_CELLS and 0 <= nj < NO_OF_CELLS:
                neighbor = self.grid[ni][nj]
                neighbors.append(neighbor)
                logging.debug(f"{name} neighbor {neighbor} at ({ni}, {nj}) added.")

        logging.debug(f"Total neighbors found for Node({i}, {j}): {len(neighbors)}")
        return neighbors

