from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContextRule(_messages.Message):
    """A context rule provides information about the context for an individual
  API element.

  Fields:
    provided: A list of full type names of provided contexts.
    requested: A list of full type names of requested contexts.
    selector: Selects the methods to which this rule applies.  Refer to
      selector for syntax details.
  """
    provided = _messages.StringField(1, repeated=True)
    requested = _messages.StringField(2, repeated=True)
    selector = _messages.StringField(3)