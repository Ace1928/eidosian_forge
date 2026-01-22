import abc
import collections
import itertools
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
def _validate_default(self, context):
    if self.default is not None:
        default_value = self.default
        if self.type == self.LIST and (not isinstance(self.default, list)):
            try:
                default_value = self.default.split(',')
            except (KeyError, AttributeError) as err:
                raise exception.InvalidSchemaError(message=_('Default must be a comma-delimited list string: %s') % err)
        elif self.type == self.LIST and isinstance(self.default, list):
            default_value = [str(x) for x in self.default]
        try:
            self.validate_constraints(default_value, context, [constr.CustomConstraint])
        except (ValueError, TypeError, exception.StackValidationFailed) as exc:
            raise exception.InvalidSchemaError(message=_('Invalid default %(default)s (%(exc)s)') % dict(default=self.default, exc=exc))