from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendRule(_messages.Message):
    """A backend rule provides configuration for an individual API element.

  Fields:
    address: The address of the API backend.
    deadline: The number of seconds to wait for a response from a request.
      The default depends on the deployment context.
    selector: Selects the methods to which this rule applies.  Refer to
      selector for syntax details.
  """
    address = _messages.StringField(1)
    deadline = _messages.FloatField(2)
    selector = _messages.StringField(3)