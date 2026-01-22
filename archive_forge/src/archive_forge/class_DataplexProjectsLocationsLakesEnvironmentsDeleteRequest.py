from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesEnvironmentsDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesEnvironmentsDeleteRequest object.

  Fields:
    name: Required. The resource name of the environment: projects/{project_id
      }/locations/{location_id}/lakes/{lake_id}/environments/{environment_id}.
  """
    name = _messages.StringField(1, required=True)