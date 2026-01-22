from Algorithm import Algorithm
from typing import List, Optional
import logging

# To avoid side effects of logging configuration on other modules, we ensure that logging is configured
# only when the module is executed as the main module. This encapsulation prevents the logging configurations
# from affecting other modules that might import this module.
if __name__ == "__main__":
    # Configure logging to debug level and specify the format for logging.
    # This configuration is critical for tracing the execution of the algorithm and understanding its flow.
    # The format includes the time of the log entry, the level of severity, and the message.
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class A_STAR(Algorithm):
    def __init__(self, grid: "Grid") -> None:
        # The constructor of the A_STAR class initializes the algorithm with a specific grid.
        # It inherits from the Algorithm class, allowing it to use shared functionality and structure.
        super().__init__(grid)

        # A debug log entry is created when an instance of A_STAR is initialized.
        # This log entry will help in debugging by confirming that the grid has been successfully initialized.
        logging.debug("A_STAR algorithm initialized with grid.")

    def run_algorithm(self, snake: "Snake") -> Optional[List["Node"]]:
        """
        Executes the A* algorithm to find the shortest path from the snake's current position to the goal state.

        This method initializes the search space, sets up the initial and goal states, and processes each node
        until the goal is reached or the search space is exhausted. It uses a list to manage the frontier nodes
        and another to keep track of explored nodes, ensuring that no node is processed more than once.

        Args:
            snake (Snake): The snake object containing the current state of the snake in the game.

        Returns:
            Optional[List["Node"]]: A list of Node objects representing the path from the start to the goal state.
            If no path is found, it returns None.
        """
        # Clear all the previous state data to ensure a fresh start for the algorithm.
        self.frontier: List["Node"] = []  # List to store the nodes yet to be explored.
        self.explored_set: List["Node"] = (
            []
        )  # List to store the nodes that have been explored.
        self.path: List["Node"] = (
            []
        )  # List to store the path from the start to the goal node.

        # Attempt to retrieve the initial and goal states from the provided snake object.
        # These states are essential for guiding the search process of the A* algorithm.
        try:
            initialstate: "Node"
            goalstate: "Node"
            initialstate, goalstate = self.get_initstate_and_goalstate(snake)
        except Exception as e:
            # Log any exceptions raised during the initialization of states, providing a debug trace.
            logging.error(f"Failed to initialize states in A* algorithm: {str(e)}")
            return None

        # Add the initial state to the frontier list to kickstart the algorithm.
        # This is the first node from which the algorithm will begin its search.
        self.frontier.append(initialstate)
        # Log the addition of the initial state to the frontier for debugging purposes.
        logging.debug(f"Initial state {initialstate} added to frontier.")

        # The following loop continues as long as there are nodes in the frontier list.
        # The frontier list contains nodes that are yet to be explored.
        while len(self.frontier) > 0:
            # Initialize the index of the node with the lowest f(n) value.
            # f(n) is the sum of g(n) (the cost from the start node to n) and h(n) (the heuristic estimate from n to the goal).
            lowest_index: int = 0

            # Iterate through the frontier list to find the node with the lowest f(n) value.
            for i in range(len(self.frontier)):
                # If the f value of the current node in the frontier is less than the f value of the node at lowest_index,
                # update lowest_index to the current index.
                if self.frontier[i].f < self.frontier[lowest_index].f:
                    lowest_index = i

            # Remove the node with the lowest f(n) value from the frontier list.
            # This node will be processed to explore its neighbors.
            lowest_node: "Node" = self.frontier.pop(lowest_index)

            # Log the removal of the node with the lowest f(n) value for debugging purposes.
            # This helps in tracing the algorithm's progress and understanding the sequence of node exploration.
            logging.debug(f"Lowest node {lowest_node} popped from frontier.")
            # check if it's goal state
            if lowest_node.equal(goalstate):
                logging.info("Goal state reached.")
                return self.get_path(lowest_node)

            self.explored_set.append(lowest_node)  # mark visited
            logging.debug(f"Node {lowest_node} added to explored set.")
            neighbors: List["Node"] = self.get_neighbors(lowest_node)  # get neighbors

            # for each neighbor
            for neighbor in neighbors:
                # check if path inside snake, outside boundary or already visited
                if (
                    self.inside_body(snake, neighbor)
                    or self.outside_boundary(neighbor)
                    or neighbor in self.explored_set
                ):
                    logging.debug(
                        f"Skipping neighbor {neighbor} due to invalid conditions."
                    )
                    continue  # skip this path

                g: int = lowest_node.g + 1
                best: bool = False  # assuming neighbor path is better

                if neighbor not in self.frontier:  # first time visiting
                    neighbor.h = self.manhattan_distance(goalstate, neighbor)
                    self.frontier.append(neighbor)
                    best = True
                    logging.debug(
                        f"Neighbor {neighbor} added to frontier with heuristic {neighbor.h}."
                    )
                elif lowest_node.g < neighbor.g:  # has already been visited
                    best = True  # but had a worse g now its better

                if best:
                    neighbor.parent = lowest_node
                    neighbor.g = g
                    neighbor.f = neighbor.g + neighbor.h
                    logging.debug(f"Updated neighbor {neighbor} with new g, f values.")
        logging.info("No path found to goal state.")
        return None
