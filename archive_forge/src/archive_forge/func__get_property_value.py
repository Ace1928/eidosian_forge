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
def _get_property_value(self, key, validate=False):
    if key not in self:
        raise KeyError(_('Invalid Property %s') % key)
    prop = self.props[key]
    value, found = self._resolve_user_value(key, prop, validate)
    if found:
        return value
    if self.translation.has_translation(prop.path):
        value = self.translation.translate(prop.path, prop_data=self.data, validate=validate)
        if value is not None or prop.has_default():
            return prop.get_value(value)
    if prop.has_default():
        return prop.get_value(None, validate, translation=self.translation)
    elif prop.required():
        raise ValueError(_('Property %s not assigned') % key)
    elif key == 'description' and prop.schema.update_allowed:
        return self.rsrc_description
    else:
        return None