from typing import Dict, List
from nbformat import NotebookNode
class CellControlSignal(Exception):
    """
    A custom exception used to indicate that the exception is used for cell
    control actions (not the best model, but it's needed to cover existing
    behavior without major refactors).
    """
    pass