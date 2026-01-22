from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsServicesGetRequest(_messages.Message):
    """A RunProjectsLocationsServicesGetRequest object.

  Fields:
    name: Required. The full name of the Service. Format:
      projects/{project}/locations/{location}/services/{service}, where
      {project} can be project id or number.
  """
    name = _messages.StringField(1, required=True)