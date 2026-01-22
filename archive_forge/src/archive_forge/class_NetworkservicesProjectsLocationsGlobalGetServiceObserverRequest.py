from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsGlobalGetServiceObserverRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsGlobalGetServiceObserverRequest
  object.

  Fields:
    name: Required. A name of the ServiceObserver to get. Must be in the
      format `projects/*/locations/global/serviceObserver`.
  """
    name = _messages.StringField(1, required=True)