class DumbMatrix(object):

    def __init__(self, value):
        self.value = value

    def __matmul__(self, other):
        if isinstance(other, DumbMatrix):
            return DumbMatrix(self.value * other.value)
        return NotImplemented

    def __imatmul__(self, other):
        if isinstance(other, DumbMatrix):
            self.value *= other.value
            return self
        return NotImplemented