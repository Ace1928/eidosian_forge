from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
def AsValidator(validator):
    """Wrap various types as instances of a validator.

  Used to allow shorthand for common validator types.  It
  converts the following types to the following Validators.

    strings -> Regex
    type -> Type
    collection -> Options
    Validator -> Its self!

  Args:
    validator: Object to wrap in a validator.

  Returns:
    Validator instance that wraps the given value.

  Raises:
    AttributeDefinitionError: if validator is not one of the above described
      types.
  """
    if six_subset.is_basestring(validator) or validator == six_subset.string_types:
        return StringValidator()
    if isinstance(validator, (str, six_subset.text_type)):
        return Regex(validator, type(validator))
    if isinstance(validator, type):
        return Type(validator)
    if isinstance(validator, (list, tuple, set)):
        return Options(*tuple(validator))
    if isinstance(validator, Validator):
        return validator
    else:
        raise AttributeDefinitionError('%s is not a valid validator' % str(validator))