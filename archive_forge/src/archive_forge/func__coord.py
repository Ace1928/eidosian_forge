import warnings
def _coord(self, lineno, column=None):
    return Coord(file=self.clex.filename, line=lineno, column=column)