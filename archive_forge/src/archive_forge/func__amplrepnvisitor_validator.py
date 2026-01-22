import enum
from pyomo.common.config import ConfigDict, ConfigValue, InEnum
from pyomo.common.modeling import NOTSET
from pyomo.repn.plugins.nl_writer import AMPLRepnVisitor, text_nl_template
from pyomo.repn.util import FileDeterminism, FileDeterminism_to_SortComponents
def _amplrepnvisitor_validator(visitor):
    if not isinstance(visitor, AMPLRepnVisitor):
        raise TypeError("'visitor' config argument should be an instance of AMPLRepnVisitor")
    return visitor