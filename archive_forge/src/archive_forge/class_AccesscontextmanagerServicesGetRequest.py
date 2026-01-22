from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerServicesGetRequest(_messages.Message):
    """A AccesscontextmanagerServicesGetRequest object.

  Fields:
    name: The name of the service to get information about. The names must be
      in the same format as used in defining a service perimeter, for example,
      `storage.googleapis.com`.
  """
    name = _messages.StringField(1, required=True)