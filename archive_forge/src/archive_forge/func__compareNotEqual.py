from io import StringIO
def _compareNotEqual(self, elem):
    return self.lhs.value(elem) != self.rhs.value(elem)