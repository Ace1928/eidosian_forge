from __future__ import absolute_import
import logging
from googlecloudsdk.third_party.appengine.admin.tools.conversion import converters
class SchemaField(object):
    """Transformation strategy from input dictionary to an output dictionary.

  Each subclass defines a different strategy for how an input value is converted
  to an output value. ConvertValue() makes a copy of the input with the proper
  transformations applied. Additionally, constraints about the input structure
  are validated while doing the transformation.
  """

    def __init__(self, target_name=None, converter=None):
        """Constructor.

    Args:
      target_name: New field name to use when creating an output dictionary. If
        None is specified, then the original name is used.
      converter: A function which performs a transformation on the value of the
        field.
    """
        self.target_name = target_name
        self.converter = converter

    def ConvertValue(self, value):
        """Convert an input value using the given schema and converter.

    This method is not meant to be overwritten. Update _VisitInternal to change
    the behavior.

    Args:
      value: Input value.

    Returns:
      Output which has been transformed using the given schema for renaming and
      converter, if specified.
    """
        result = self._VisitInternal(value)
        return self._PerformConversion(result)

    def _VisitInternal(self, value):
        """Shuffles the input value using the renames specified in the schema.

    Only structural changes are made (e.g. renaming keys, copying lists, etc.).
    Subclasses are expected to override this.

    Args:
      value: Input value.

    Returns:
      Output which has been transformed using the given schema.
    """
        raise NotImplementedError()

    def _PerformConversion(self, result):
        """Transforms the result value if a converter is specified."""
        return self.converter(result) if self.converter else result