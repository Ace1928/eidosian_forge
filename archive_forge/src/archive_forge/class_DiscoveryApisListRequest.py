from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DiscoveryApisListRequest(_messages.Message):
    """A DiscoveryApisListRequest object.

  Enums:
    LabelValueValuesEnum: Only include APIs with a matching label, such as
      'graduated' or 'labs'.

  Fields:
    label: Only include APIs with a matching label, such as 'graduated' or
      'labs'.
    name: Only include APIs with the given name.
    preferred: Return only the preferred version of an API.
  """

    class LabelValueValuesEnum(_messages.Enum):
        """Only include APIs with a matching label, such as 'graduated' or
    'labs'.

    Values:
      deprecated: APIs that have been deprecated.
      graduated: Supported APIs that have graduated from labs.
      labs: APIs that are experimental
    """
        deprecated = 0
        graduated = 1
        labs = 2
    label = _messages.EnumField('LabelValueValuesEnum', 1)
    name = _messages.StringField(2)
    preferred = _messages.BooleanField(3, default=False)