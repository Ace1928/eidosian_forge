import re
from referencing.jsonschema import lookup_recursive_ref
from jsonschema import _utils
from jsonschema.exceptions import ValidationError
def unevaluatedProperties_draft2019(validator, uP, instance, schema):
    if not validator.is_type(instance, 'object'):
        return
    evaluated_keys = find_evaluated_property_keys_by_schema(validator, instance, schema)
    unevaluated_keys = []
    for property in instance:
        if property not in evaluated_keys:
            for _ in validator.descend(instance[property], uP, path=property, schema_path=property):
                unevaluated_keys.append(property)
    if unevaluated_keys:
        if uP is False:
            error = 'Unevaluated properties are not allowed (%s %s unexpected)'
            extras = sorted(unevaluated_keys, key=str)
            yield ValidationError(error % _utils.extras_msg(extras))
        else:
            error = 'Unevaluated properties are not valid under the given schema (%s %s unevaluated and invalid)'
            yield ValidationError(error % _utils.extras_msg(unevaluated_keys))