from Constants import NO_OF_CELLS


class Node:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        self.h = 0
        self.g = 0
        self.f = 1000000
        self.parent = None

    def __str__(self):
        return f"Node(x: {self.x}, y: {self.y}, h: {self.h}, g: {self.g}, f: {self.f})"

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return (
            self.x == other.x
            and self.y == other.y
            and self.h == other.h
            and self.g == other.g
            and self.f == other.f
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return (self.f, self.h, self.g, self.x, self.y) < (
            other.f,
            other.h,
            other.g,
            other.x,
            other.y,
        )

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return (self.f, self.h, self.g, self.x, self.y) > (
            other.f,
            other.h,
            other.g,
            other.x,
            other.y,
        )

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __hash__(self):
        return hash((self.x, self.y, self.h, self.g, self.f))

    def print(self):
        print(self.__str__())

    def equal(self, other):
        return self.__eq__(other)


class Grid:
    def __init__(self):
        self.grid = []

        for i in range(NO_OF_CELLS):
            col = []
            for j in range(NO_OF_CELLS):
                col.append(Node(i, j))
            self.grid.append(col)
