from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class FormatMessage(_messages.Message):
    """Represents a message with parameters.

  Fields:
    format: Format template for the message. The `format` uses placeholders
      `$0`, `$1`, etc. to reference parameters. `$$` can be used to denote the
      `$` character. Examples: * `Failed to load '$0' which helps debug $1 the
      first time it is loaded. Again, $0 is very important.` * `Please pay
      $$10 to use $0 instead of $1.`
    parameters: Optional parameters to be embedded into the message.
  """
    format = _messages.StringField(1)
    parameters = _messages.StringField(2, repeated=True)