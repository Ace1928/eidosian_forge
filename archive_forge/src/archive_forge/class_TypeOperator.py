import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
class TypeOperator(object):
    """An n-ary type constructor which builds a new type from old"""

    def __init__(self, name, types):
        self.name = name
        self.types = types

    def __str__(self):
        num_types = len(self.types)
        if num_types == 0:
            return self.name
        elif self.name == 'fun':
            return 'Callable[[{0}], {1}]'.format(', '.join(map(str, self.types[:-1])), self.types[-1])
        elif self.name == 'option':
            return 'Option[{0}]'.format(self.types[0])
        else:
            return '{0}[{1}]'.format(self.name.capitalize(), ', '.join(map(str, self.types)))