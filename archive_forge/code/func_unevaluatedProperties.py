from fractions import Fraction
import re
from jsonschema._utils import (
from jsonschema.exceptions import FormatError, ValidationError
def unevaluatedProperties(validator, unevaluatedProperties, instance, schema):
    if not validator.is_type(instance, 'object'):
        return
    evaluated_keys = find_evaluated_property_keys_by_schema(validator, instance, schema)
    unevaluated_keys = []
    for property in instance:
        if property not in evaluated_keys:
            for _ in validator.descend(instance[property], unevaluatedProperties, path=property, schema_path=property):
                unevaluated_keys.append(property)
    if unevaluated_keys:
        if unevaluatedProperties is False:
            error = 'Unevaluated properties are not allowed (%s %s unexpected)'
            extras = sorted(unevaluated_keys, key=str)
            yield ValidationError(error % extras_msg(extras))
        else:
            error = 'Unevaluated properties are not valid under the given schema (%s %s unevaluated and invalid)'
            yield ValidationError(error % extras_msg(unevaluated_keys))