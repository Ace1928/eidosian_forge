from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsServicesListRequest(_messages.Message):
    """A RunProjectsLocationsServicesListRequest object.

  Fields:
    pageSize: Maximum number of Services to return in this call.
    pageToken: A page token received from a previous call to ListServices. All
      other parameters must match.
    parent: Required. The location and project to list resources on. Location
      must be a valid Google Cloud region, and cannot be the "-" wildcard.
      Format: projects/{project}/locations/{location}, where {project} can be
      project id or number.
    showDeleted: If true, returns deleted (but unexpired) resources along with
      active ones.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)