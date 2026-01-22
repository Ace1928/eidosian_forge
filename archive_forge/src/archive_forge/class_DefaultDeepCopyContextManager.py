import abc
import copy
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.utilities import performance_utils as pu
from cvxpy.utilities.deterministic import unique_list
class DefaultDeepCopyContextManager:
    """
    override custom __deepcopy__ implementation and call copy.deepcopy's implementation instead
    """

    def __init__(self, item):
        self.item = item
        self.deepcopy = None

    def __enter__(self):
        self.deepcopy = getattr(self.item, '__deepcopy__', _MISSING)
        if self.deepcopy is not _MISSING:
            self.item.__deepcopy__ = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.deepcopy is not _MISSING:
            self.item.__deepcopy__ = self.deepcopy
            self.deepcopy = _MISSING