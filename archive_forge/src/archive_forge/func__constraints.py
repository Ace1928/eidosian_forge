from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import parameters
@classmethod
def _constraints(cls, param_name, schema_dict):
    constraints = schema_dict.get(cls.CONSTRAINTS)
    if constraints is None:
        return
    if not isinstance(constraints, list):
        raise exception.InvalidSchemaError(message=_('Invalid parameter constraints for parameter %s, expected a list') % param_name)
    for constraint in constraints:
        cls._check_dict(constraint, PARAM_CONSTRAINTS, 'parameter constraints')
        yield cls._constraint_from_def(constraint)