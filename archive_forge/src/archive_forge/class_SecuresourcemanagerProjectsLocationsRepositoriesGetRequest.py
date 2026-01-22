from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuresourcemanagerProjectsLocationsRepositoriesGetRequest(_messages.Message):
    """A SecuresourcemanagerProjectsLocationsRepositoriesGetRequest object.

  Fields:
    name: Required. Name of the repository to retrieve. The format is `project
      s/{project_number}/locations/{location_id}/repositories/{repository_id}`
      .
  """
    name = _messages.StringField(1, required=True)