import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def uniqueItems(validator, uI, instance, schema):
    if uI and validator.is_type(instance, 'array') and (not _utils.uniq(instance)):
        yield ValidationError('%r has non-unique elements' % (instance,))