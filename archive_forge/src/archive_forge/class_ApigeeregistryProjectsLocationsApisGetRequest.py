from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisGetRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisGetRequest object.

  Fields:
    name: Required. The name of the API to retrieve. Format:
      `projects/*/locations/*/apis/*`
  """
    name = _messages.StringField(1, required=True)