from Constants import NO_OF_CELLS


class Node:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        self.h = 0  # Heuristic cost to the goal (not used initially)
        self.g = 0  # Cost from the start node
        self.f = 1000000  # Initially set to a high value to represent an unvisited node
        self.parent = None  # Parent node in the path

    def __str__(self):
        # Provides a string representation of the Node, useful for debugging and logging
        return f"Node(x: {self.x}, y: {self.y}, h: {self.h}, g: {self.g}, f: {self.f})"

    def __eq__(self, other):
        # Checks equality based on the position of the nodes only
        if not isinstance(other, Node):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        # Less than comparison based on f value for priority queue operations in pathfinding algorithms
        if not isinstance(other, Node):
            return NotImplemented
        return self.f < other.f

    def __gt__(self, other):
        # Greater than comparison based on f value for priority queue operations in pathfinding algorithms
        if not isinstance(other, Node):
            return NotImplemented
        return self.f > other.f

    def __le__(self, other):
        # Less than or equal to comparison based on f value
        if not isinstance(other, Node):
            return NotImplemented
        return self.f <= other.f

    def __ge__(self, other):
        # Greater than or equal to comparison based on f value
        if not isinstance(other, Node):
            return NotImplemented
        return self.f >= other.f


class Grid:
    def __init__(self):
        self.grid = []

        for i in range(NO_OF_CELLS):
            col = []
            for j in range(NO_OF_CELLS):
                col.append(Node(i, j))
            self.grid.append(col)
