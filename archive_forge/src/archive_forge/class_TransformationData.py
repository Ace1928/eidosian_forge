from pyomo.common import Factory
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import MouseTrap
from pyomo.common.deprecation import deprecated
from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import TransformationTimer
class TransformationData(object):
    """
    This is a container class that supports named data objects.
    """

    def __init__(self):
        self._data = {}

    def __getitem__(self, name):
        if not name in self._data:
            self._data[name] = TransformationInfo()
        return self._data[name]