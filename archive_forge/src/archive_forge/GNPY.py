from Utility import Node
from Algorithm import Algorithm
import logging
import asyncio

# Configure logging to the most detailed level possible
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DFS(Algorithm):
    """
    Depth-First Search (DFS) algorithm for pathfinding in a grid-based snake game.

    Attributes:
        grid (list): A grid representation of the game environment.
    """

    def __init__(self, grid: list):
        """
        Initialize the DFS algorithm with the provided grid.

        Args:
            grid (list): The grid representing the game environment.
        """
        super().__init__(grid)
        logging.debug("DFS algorithm instance initialized with grid.")

    async def recursive_DFS(self, snake, goalstate: Node, currentstate: Node) -> Node:
        """
        Recursively explore the grid to find a path from the current state to the goal state using DFS.

        Args:
            snake: The snake object, used to check collisions with the snake's body.
            goalstate (Node): The goal state node.
            currentstate (Node): The current state node being explored.

        Returns:
            Node: The node representing the path to the goal if found, otherwise None.
        """
        logging.debug(
            f"Entering recursive DFS with current state: {currentstate} and goal state: {goalstate}"
        )

        # Check if the current state is the goal state
        if currentstate == goalstate:
            logging.debug("Goal state reached in recursive DFS.")
            return self.get_path(currentstate)

        # Check if the current state has already been visited
        if currentstate in self.explored_set:
            logging.debug(f"State {currentstate} already explored.")
            return None

        # Mark the current state as visited
        self.explored_set.append(currentstate)
        logging.debug(f"State {currentstate} added to explored set.")

        # Retrieve neighbors of the current state
        neighbors = self.get_neighbors(currentstate)
        logging.debug(f"Neighbors retrieved: {neighbors}")

        # Iterate through each neighbor
        for neighbor in neighbors:
            if (
                not self.inside_body(snake, neighbor)
                and not self.outside_boundary(neighbor)
                and neighbor not in self.explored_set
            ):
                neighbor.parent = currentstate  # Mark parent node
                logging.debug(
                    f"Exploring neighbor {neighbor} with parent set to {currentstate}"
                )
                path = await self.recursive_DFS(
                    snake, goalstate, neighbor
                )  # Recursive call to explore the neighbor
                if path is not None:
                    logging.debug(f"Path found: {path}")
                    return path  # Path found
        logging.debug("No path found, returning None.")
        return None

    async def run_algorithm(self, snake) -> Node:
        """
        Execute the DFS algorithm to find a path for the snake.

        Args:
            snake: The snake object for which the path is being calculated.

        Returns:
            Node: The next node in the path for the snake to follow, if any.
        """
        logging.debug("Running DFS algorithm.")

        # To avoid looping in the same location
        if len(self.path) != 0:
            # while you have path keep going
            path = self.path.pop()
            logging.debug(f"Path popped from path list: {path}")

            if self.inside_body(snake, path):
                self.path = []  # Clear path if it leads into the snake's body
                logging.debug("Path leads into snake's body. Path list cleared.")
            else:
                logging.debug(f"Returning path: {path}")
                return path

        # Start with a clean state
        self.frontier = []
        self.explored_set = []
        self.path = []
        logging.debug("Frontier, explored set, and path lists cleared.")

        # Initialize the initial and goal states
        initialstate, goalstate = self.get_initstate_and_goalstate(snake)
        logging.debug(f"Initial state: {initialstate}, Goal state: {goalstate}")

        # Append the initial state to the frontier
        self.frontier.append(initialstate)
        logging.debug(f"Initial state {initialstate} added to frontier.")

        # Start the recursive DFS
        result_path = await self.recursive_DFS(snake, goalstate, initialstate)
        logging.debug(f"DFS completed with result path: {result_path}")
        return result_path
