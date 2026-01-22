from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1VpcAccessibleServices(_messages.Message):
    """Specifies how APIs are allowed to communicate within the Service
  Perimeter.

  Fields:
    allowedServices: The list of APIs usable within the Service Perimeter.
      Must be empty unless 'enable_restriction' is True. You can specify a
      list of individual services, as well as include the 'RESTRICTED-
      SERVICES' value, which automatically includes all of the services
      protected by the perimeter.
    enableRestriction: Whether to restrict API calls within the Service
      Perimeter to the list of APIs specified in 'allowed_services'.
  """
    allowedServices = _messages.StringField(1, repeated=True)
    enableRestriction = _messages.BooleanField(2)