import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
def _print_tree(self):
    """
        Print a text representation of the node tree, which is useful for
        debugging purposes.
        """
    self._cpp_trifinder.print_tree()