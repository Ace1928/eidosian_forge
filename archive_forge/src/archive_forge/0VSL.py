from Algorithm import Algorithm
from typing import List, Optional, Dict, Any
import logging
import heapq

# Constants for readability and maintainability
STEP_COST = 1

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


class Node:
    # A data class that represents a node in the grid.
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.parent = None
        self.g = 0  # Cost from start node
        self.h = 0  # Heuristic to goal node
        self.f = 0  # Total cost

    def __lt__(self, other: "Node"):
        # This method makes Node compatible with heapq by comparing the f value.
        return self.f < other.f


class A_STAR(Algorithm):
    def __init__(self, grid: "Grid") -> None:
        # The constructor of the A_STAR class initializes the algorithm with a specific grid.
        # It inherits from the Algorithm class, allowing it to use shared functionality and structure.
        super().__init__(grid)

        # A debug log entry is created when an instance of A_STAR is initialized.
        # This log entry will help in debugging by confirming that the grid has been successfully initialized.
        logging.debug("A_STAR algorithm initialized with grid.")

    def run_algorithm(self, snake: "Snake") -> Optional[List[Node]]:
        """
        Executes the A* algorithm to find the shortest path from the snake's current position to the goal state.

        This method initializes the search space, sets up the initial and goal states, and processes each node
        until the goal is reached or the search space is exhausted. It uses a priority queue to manage the frontier nodes
        and another list to keep track of explored nodes, ensuring that no node is processed more than once.

        Args:
            snake (Snake): The snake object containing the current state of the snake in the game.

        Returns:
            Optional[List[Node]]: A list of Node objects representing the path from the start to the goal state.
            If no path is found, it returns None.
        """
        # Clear all the previous state data to ensure a fresh start for the algorithm.
        self.frontier = []  # Priority queue to store the nodes yet to be explored.
        heapq.heapify(self.frontier)
        self.explored_set = []  # List to store the nodes that have been explored.
        self.path = []  # List to store the path from the start to the goal node.

        # Attempt to retrieve the initial and goal states from the provided snake object.
        # These states are essential for guiding the search process of the A* algorithm.
        try:
            initialstate: Node
            goalstate: Node
            initialstate, goalstate = self.get_initstate_and_goalstate(snake)
        except Exception as e:
            # Log any exceptions raised during the initialization of states, providing a debug trace.
            logging.error(f"Failed to initialize states in A* algorithm: {str(e)}")
            return None

        # Add the initial state to the frontier list to kickstart the algorithm.
        # This is the first node from which the algorithm will begin its search.
        heapq.heappush(self.frontier, initialstate)
        # Log the addition of the initial state to the frontier for debugging purposes.
        logging.debug(f"Initial state {initialstate} added to frontier.")

        # The following loop continues as long as there are nodes in the frontier list.
        # The frontier list contains nodes that are yet to be explored.
        while self.frontier:
            # Remove the node with the lowest f(n) value from the frontier list.
            # This node will be processed to explore its neighbors.
            lowest_node: Node = heapq.heappop(self.frontier)

            # Log the removal of the node with the lowest f(n) value for debugging purposes.
            # This helps in tracing the algorithm's progress and understanding the sequence of node exploration.
            logging.debug(f"Lowest node {lowest_node} popped from frontier.")
            # The following block checks if the current node being processed, `lowest_node`, is the goal state.
            # If `lowest_node` is equivalent to `goalstate`, it logs that the goal state has been reached and returns the path.
            try:
                if lowest_node.equal(goalstate):
                    logging.info("Goal state reached.")
                    # Retrieve the path from the start node to the goal node using the `get_path` method.
                    # This method traces back from the goal node to the start node using node parents.
                    path_to_goal = self.get_path(lowest_node)
                    return path_to_goal
            except Exception as e:
                # Log any exceptions that occur during the goal state check or path retrieval.
                # This ensures that all errors are captured and can be debugged effectively.
                logging.error(f"Error checking goal state or retrieving path: {str(e)}")
                return None

            # The current node, `lowest_node`, is added to the `explored_set` to mark it as visited.
            # This prevents the algorithm from processing this node again, ensuring efficiency.
            self.explored_set.append(lowest_node)
            # Log the addition of `lowest_node` to the explored set for debugging and traceability.
            logging.debug(f"Node {lowest_node} added to explored set.")

            # Retrieve the neighboring nodes of `lowest_node` using the `get_neighbors` method.
            # This method calculates adjacent nodes based on the current node's position.
            try:
                neighbors: List[Node] = self.get_neighbors(lowest_node)
            except Exception as e:
                # Log any exceptions that occur during the retrieval of neighbors.
                # This ensures that errors in neighbor calculation are captured for debugging.
                logging.error(
                    f"Error retrieving neighbors for node {lowest_node}: {str(e)}"
                )
                neighbors = (
                    []
                )  # Continue with an empty list of neighbors if an error occurs.

            # This loop iterates over each neighbor node of the current node being processed, `lowest_node`.
            for neighbor in neighbors:
                # The following conditional checks are performed to determine if the neighbor node should be skipped:
                # 1. `self.inside_body(snake, neighbor)`: Checks if the neighbor node is inside the snake's body.
                # 2. `self.outside_boundary(neighbor)`: Checks if the neighbor node is outside the defined game boundary.
                # 3. `neighbor in self.explored_set`: Checks if the neighbor node has already been explored.
                # These checks ensure that the algorithm does not process invalid or redundant nodes, maintaining efficiency.
                if (
                    self.inside_body(
                        snake, neighbor
                    )  # Check if the neighbor is inside the snake's body.
                    or self.outside_boundary(
                        neighbor
                    )  # Check if the neighbor is outside the game boundary.
                    or neighbor
                    in self.explored_set  # Check if the neighbor has already been explored.
                ):
                    # If any of the above conditions are true, the neighbor node is skipped.
                    # A debug log entry is created to record the skipping of this neighbor node.
                    # This logging is crucial for debugging and understanding the flow of node processing.
                    logging.debug(
                        f"Skipping neighbor {neighbor} due to invalid conditions."
                    )
                    continue  # The `continue` statement skips the current iteration and moves to the next neighbor.

                # The variable `g` represents the tentative cost from the start node to the neighbor node through `lowest_node`.
                # It is calculated as the g-value of `lowest_node` incremented by STEP_COST, assuming each step has a uniform cost.
                g: int = (
                    lowest_node.g + STEP_COST
                )  # Tentative g-value for the neighbor node.

                # The variable `best` is a boolean flag used to indicate whether the path through `lowest_node` to `neighbor`
                # is currently considered the best known path. It is initially set to False and may be updated later in the algorithm.
                best: bool = False  # Initialize the best path flag as False.

                # First, we check if the neighbor node has not been previously added to the frontier.
                # This condition implies that the neighbor node is being visited for the first time.
                if neighbor not in self.frontier:
                    # Calculate the heuristic value (h) for the neighbor node using the Manhattan distance
                    # from the neighbor node to the goal state. This heuristic is crucial for the A* algorithm's
                    # performance as it influences the order of node exploration based on estimated cost to the goal.
                    neighbor.h = self.manhattan_distance(goalstate, neighbor)

                    # Add the newly evaluated neighbor node to the frontier list.
                    # The frontier serves as the collection of nodes that are pending exploration.
                    heapq.heappush(self.frontier, neighbor)

                    # Set the 'best' flag to True indicating that the current path to this neighbor node
                    # is the best one found so far. This flag helps in determining whether to update the node's
                    # parent and cost values later in the algorithm.
                    best = True

                    # Log the addition of the neighbor node to the frontier along with its heuristic value.
                    # This logging is essential for debugging and understanding the sequence of node additions
                    # to the frontier, which directly affects the algorithm's pathfinding process.
                    logging.debug(
                        f"Neighbor {neighbor} added to frontier with heuristic {neighbor.h}."
                    )
                else:
                    # If the neighbor node is already in the frontier, we then check if the current path to the neighbor
                    # (through the current lowest_node) offers a lower g value (cost from start node to this neighbor)
                    # than previously known. If so, this path is more efficient.
                    for i in range(len(self.frontier)):
                        if self.frontier[i] == neighbor:
                            if g < self.frontier[i].g:
                                # Since a more efficient path is found, set the 'best' flag to True.
                                # This indicates that the neighbor node's parent and cost values need updating.
                                best = True  # but had a worse g now its better
                                break

                # If the 'best' flag is set to True, it means either a new node has been added to the frontier
                # or a more efficient path has been found for an existing node in the frontier.
                if best:
                    # Update the parent of the neighbor node to the current lowest_node.
                    # This linkage is crucial for tracing the path back from the goal node to the start node
                    # once the goal is reached.
                    neighbor.parent = lowest_node

                    # Update the g value (cost from start node to this neighbor) of the neighbor node.
                    neighbor.g = g

                    # Calculate the f value of the neighbor node, which is the sum of its g value and h value.
                    # The f value represents the total estimated cost of the path through this node to the goal.
                    neighbor.f = neighbor.g + neighbor.h

                    # Log the update of the neighbor node's g and f values.
                    # This logging is vital for debugging and verifying that the node's values are updated correctly
                    # when a better path is found.
                    logging.debug(f"Updated neighbor {neighbor} with new g, f values.")

        # If the frontier is exhausted and no path to the goal state is found, log this outcome.
        # This information is crucial for understanding whether the algorithm is functioning as expected
        # and if the goal state is reachable from the start state under the given conditions.
        logging.info("No path found to goal state.")

        # Return None to indicate that no path could be found.
        # This return value is essential for the caller to handle the case where no path is available.
        return None
