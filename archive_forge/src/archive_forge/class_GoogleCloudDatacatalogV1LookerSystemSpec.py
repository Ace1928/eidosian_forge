from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1LookerSystemSpec(_messages.Message):
    """Specification that applies to entries that are part `LOOKER` system
  (user_specified_type)

  Fields:
    parentInstanceDisplayName: Name of the parent Looker Instance. Empty if it
      does not exist.
    parentInstanceId: ID of the parent Looker Instance. Empty if it does not
      exist. Example value: `someinstance.looker.com`
    parentModelDisplayName: Name of the parent Model. Empty if it does not
      exist.
    parentModelId: ID of the parent Model. Empty if it does not exist.
    parentViewDisplayName: Name of the parent View. Empty if it does not
      exist.
    parentViewId: ID of the parent View. Empty if it does not exist.
  """
    parentInstanceDisplayName = _messages.StringField(1)
    parentInstanceId = _messages.StringField(2)
    parentModelDisplayName = _messages.StringField(3)
    parentModelId = _messages.StringField(4)
    parentViewDisplayName = _messages.StringField(5)
    parentViewId = _messages.StringField(6)