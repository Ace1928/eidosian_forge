from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DatabaseFlags(_messages.Message):
    """Database flags for Cloud SQL instances.

  Fields:
    name: The name of the flag. These flags are passed at instance startup, so
      include both server options and system variables. Flags are specified
      with underscores, not hyphens. For more information, see [Configuring
      Database Flags](https://cloud.google.com/sql/docs/mysql/flags) in the
      Cloud SQL documentation.
    value: The value of the flag. Boolean flags are set to `on` for true and
      `off` for false. This field must be omitted if the flag doesn't take a
      value.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)