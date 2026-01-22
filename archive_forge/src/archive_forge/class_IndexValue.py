from io import StringIO
class IndexValue:

    def __init__(self, index):
        self.index = int(index) - 1

    def value(self, elem):
        return elem.children[self.index]