from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
class ISOS(ICategorizedObject):
    """
    The interface for Special Ordered Sets.
    """
    __slots__ = ()
    variables = _abstract_readwrite_property(doc='The sos variables')
    weights = _abstract_readwrite_property(doc='The sos variables')
    level = _abstract_readwrite_property(doc='The sos level (e.g., 1,2,...)')

    def items(self):
        """Iterator over the sos variables and weights as tuples"""
        return zip(self.variables, self.weights)

    def __contains__(self, v):
        """Check if the sos contains the variable v"""
        for x in self.variables:
            if id(x) == id(v):
                return True

    def __len__(self):
        """The number of members in the set"""
        return len(self.variables)