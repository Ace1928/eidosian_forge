import collections
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
from heat.engine import function
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import support
from heat.engine import translation as trans
def allowed_param_prop_type(self):
    """Return allowed type of Property Schema converted from parameter.

        Especially, when generating Schema from parameter, Integer Property
        Schema will be supplied by Number parameter.
        """
    param_type_map = {self.INTEGER: self.NUMBER, self.STRING: self.STRING, self.NUMBER: self.NUMBER, self.BOOLEAN: self.BOOLEAN, self.LIST: self.LIST, self.MAP: self.MAP}
    return param_type_map[self.type]