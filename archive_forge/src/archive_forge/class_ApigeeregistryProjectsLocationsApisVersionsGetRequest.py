from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisVersionsGetRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisVersionsGetRequest object.

  Fields:
    name: Required. The name of the version to retrieve. Format:
      `projects/*/locations/*/apis/*/versions/*`
  """
    name = _messages.StringField(1, required=True)