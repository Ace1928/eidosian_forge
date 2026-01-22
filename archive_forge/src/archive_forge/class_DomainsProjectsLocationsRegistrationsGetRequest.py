from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsGetRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsGetRequest object.

  Fields:
    name: Required. The name of the `Registration` to get, in the format
      `projects/*/locations/*/registrations/*`.
  """
    name = _messages.StringField(1, required=True)