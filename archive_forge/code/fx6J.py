from Algorithm import Algorithm
from typing import List, Optional, Dict, Any
import logging
import heapq
from sys import maxsize

# Custom type for Node attributes
NodeAttributes = Dict[str, Any]

# Constants for algorithm
STEP_COST = 1


# Configure logging within a function to avoid side effects when imported
def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )


if __name__ == "__main__":
    configure_logging()


class A_STAR(Algorithm):
    def __init__(self, grid: "Grid") -> None:
        super().__init__(grid)
        logging.debug("A_STAR algorithm initialized with grid.")

    def run_algorithm(self, snake: "Snake") -> Optional[List["Node"]]:
        # clear everything
        self.frontier: List["Node"] = []
        heapq.heapify(self.frontier)  # Using a priority queue for the frontier
        self.explored_set: List["Node"] = []
        self.path: List["Node"] = []

        initialstate: "Node"
        goalstate: "Node"
        initialstate, goalstate = self.get_initstate_and_goalstate(snake)

        # open list
        heapq.heappush(self.frontier, (initialstate.f, initialstate))
        logging.debug(f"Initial state {initialstate} added to frontier.")

        # while we have states in open list
        while self.frontier:
            # get node with lowest f(n)
            _, lowest_node = heapq.heappop(self.frontier)
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

                g: int = lowest_node.g + STEP_COST
                best: bool = False  # assuming neighbor path is better

                if neighbor not in [n for _, n in self.frontier]:  # first time visiting
                    neighbor.h = self.manhattan_distance(goalstate, neighbor)
                    heapq.heappush(self.frontier, (neighbor.f, neighbor))
                    best = True
                    logging.debug(
                        f"Neighbor {neighbor} added to frontier with heuristic {neighbor.h}."
                    )
                else:
                    # Check if the current path to neighbor is better
                    for i, (_, n) in enumerate(self.frontier):
                        if n == neighbor and g < n.g:
                            self.frontier[i] = (g, neighbor)
                            heapq.heapify(self.frontier)
                            best = True
                            break
                        elif n == neighbor and g == n.g:
                            best = True
                            del self.frontier[i]
                            heapq.heapify(self.frontier)
                            break

                if best:
                    neighbor.parent = lowest_node
                    neighbor.g = g
                    neighbor.f = neighbor.g + neighbor.h
                    logging.debug(f"Updated neighbor {neighbor} with new g, f values.")
        logging.info("No path found to goal state.")
        return None
