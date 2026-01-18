import collections
import copy
from heat.common.i18n import _
from heat.common import exception
from heat.engine import constraints
from heat.engine import parameters
from heat.engine import properties
Return True if the presence of the output indicates an error.