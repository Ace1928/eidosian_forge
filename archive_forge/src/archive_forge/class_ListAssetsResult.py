from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAssetsResult(_messages.Message):
    """Result containing the Asset and its State.

  Enums:
    StateChangeValueValuesEnum: State change of the asset between the points
      in time.

  Fields:
    asset: Asset matching the search request.
    stateChange: State change of the asset between the points in time.
  """

    class StateChangeValueValuesEnum(_messages.Enum):
        """State change of the asset between the points in time.

    Values:
      UNUSED: State change is unused, this is the canonical default for this
        enum.
      ADDED: Asset was added between the points in time.
      REMOVED: Asset was removed between the points in time.
      ACTIVE: Asset was present at both point(s) in time.
    """
        UNUSED = 0
        ADDED = 1
        REMOVED = 2
        ACTIVE = 3
    asset = _messages.MessageField('Asset', 1)
    stateChange = _messages.EnumField('StateChangeValueValuesEnum', 2)