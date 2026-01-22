from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsJobsDeleteRequest(_messages.Message):
    """A RunProjectsLocationsJobsDeleteRequest object.

  Fields:
    etag: A system-generated fingerprint for this version of the resource. May
      be used to detect modification conflict during updates.
    name: Required. The full name of the Job. Format:
      projects/{project}/locations/{location}/jobs/{job}, where {project} can
      be project id or number.
    validateOnly: Indicates that the request should be validated without
      actually deleting any resources.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)