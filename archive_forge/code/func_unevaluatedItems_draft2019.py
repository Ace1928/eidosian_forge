import re
from referencing.jsonschema import lookup_recursive_ref
from jsonschema import _utils
from jsonschema.exceptions import ValidationError
def unevaluatedItems_draft2019(validator, unevaluatedItems, instance, schema):
    if not validator.is_type(instance, 'array'):
        return
    evaluated_item_indexes = find_evaluated_item_indexes_by_schema(validator, instance, schema)
    unevaluated_items = [item for index, item in enumerate(instance) if index not in evaluated_item_indexes]
    if unevaluated_items:
        error = 'Unevaluated items are not allowed (%s %s unexpected)'
        yield ValidationError(error % _utils.extras_msg(unevaluated_items))