from .error import MarkedYAMLError
from .tokens import *
class SimpleKey:

    def __init__(self, token_number, required, index, line, column, mark):
        self.token_number = token_number
        self.required = required
        self.index = index
        self.line = line
        self.column = column
        self.mark = mark