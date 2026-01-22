from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsComplexTypesGetRequest(_messages.Message):
    """A MediaassetProjectsLocationsComplexTypesGetRequest object.

  Fields:
    name: Required. The name of the complex type to retrieve, in the following
      form: `projects/{project}/locations/{location}/complexTypes/{type}`.
  """
    name = _messages.StringField(1, required=True)