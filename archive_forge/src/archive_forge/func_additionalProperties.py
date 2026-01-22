import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def additionalProperties(validator, aP, instance, schema):
    if not validator.is_type(instance, 'object'):
        return
    extras = set(_utils.find_additional_properties(instance, schema))
    if validator.is_type(aP, 'object'):
        for extra in extras:
            for error in validator.descend(instance[extra], aP, path=extra):
                yield error
    elif not aP and extras:
        if 'patternProperties' in schema:
            patterns = sorted(schema['patternProperties'])
            if len(extras) == 1:
                verb = 'does'
            else:
                verb = 'do'
            error = '%s %s not match any of the regexes: %s' % (', '.join(map(repr, sorted(extras))), verb, ', '.join(map(repr, patterns)))
            yield ValidationError(error)
        else:
            error = 'Additional properties are not allowed (%s %s unexpected)'
            yield ValidationError(error % _utils.extras_msg(extras))