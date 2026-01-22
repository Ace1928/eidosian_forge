from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsModulesDeleteRequest(_messages.Message):
    """A MediaassetProjectsLocationsModulesDeleteRequest object.

  Fields:
    name: Required. The name of the module to delete, in the following form:
      `projects/{project}/locations/{location}/modules/{module}`.
  """
    name = _messages.StringField(1, required=True)