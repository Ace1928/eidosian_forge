from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsRuntimesListRequest(_messages.Message):
    """A CloudfunctionsProjectsLocationsRuntimesListRequest object.

  Fields:
    filter: The filter for Runtimes that match the filter expression,
      following the syntax outlined in https://google.aip.dev/160.
    parent: Required. The project and location from which the runtimes should
      be listed, specified in the format `projects/*/locations/*`
  """
    filter = _messages.StringField(1)
    parent = _messages.StringField(2, required=True)