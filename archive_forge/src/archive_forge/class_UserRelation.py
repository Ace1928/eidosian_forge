from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserRelation(_messages.Message):
    """JSON template for a relation entry.

  Fields:
    customType: Custom Type.
    type: The relation of the user. Some of the possible values are mother,
      father, sister, brother, manager, assistant, partner.
    value: The name of the relation.
  """
    customType = _messages.StringField(1)
    type = _messages.StringField(2)
    value = _messages.StringField(3)