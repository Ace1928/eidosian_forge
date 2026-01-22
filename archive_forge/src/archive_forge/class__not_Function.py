from io import StringIO
class _not_Function:

    def __init__(self):
        self.baseValue = None

    def setParams(self, baseValue):
        self.baseValue = baseValue

    def value(self, elem):
        return not self.baseValue.value(elem)