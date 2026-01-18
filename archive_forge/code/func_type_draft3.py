import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def type_draft3(validator, types, instance, schema):
    types = _utils.ensure_list(types)
    all_errors = []
    for index, type in enumerate(types):
        if type == 'any':
            return
        if validator.is_type(type, 'object'):
            errors = list(validator.descend(instance, type, schema_path=index))
            if not errors:
                return
            all_errors.extend(errors)
        elif validator.is_type(instance, type):
            return
    else:
        yield ValidationError(_utils.types_msg(instance, types), context=all_errors)