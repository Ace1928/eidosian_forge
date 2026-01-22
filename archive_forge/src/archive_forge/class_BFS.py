from collections import deque
from Utility import Node
from Algorithm import Algorithm
class BFS(Algorithm):

    def __init__(self, grid):
        super().__init__(grid)

    def run_algorithm(self, snake):
        self.frontier = deque([])
        self.explored_set = []
        self.path = []
        initialstate, goalstate = self.get_initstate_and_goalstate(snake)
        self.frontier.append(initialstate)
        while len(self.frontier) > 0:
            shallowest_node = self.frontier.popleft()
            self.explored_set.append(shallowest_node)
            neighbors = self.get_neighbors(shallowest_node)
            for neighbor in neighbors:
                if self.inside_body(snake, neighbor) or self.outside_boundary(neighbor):
                    self.explored_set.append(neighbor)
                    continue
                if neighbor not in self.frontier and neighbor not in self.explored_set:
                    neighbor.parent = shallowest_node
                    self.explored_set.append(neighbor)
                    self.frontier.append(neighbor)
                    if neighbor.equal(goalstate):
                        return self.get_path(neighbor)
        return None