from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsGetRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsGetRequest object.

  Fields:
    name: The name of the KeyRing to get.
  """
    name = _messages.StringField(1, required=True)