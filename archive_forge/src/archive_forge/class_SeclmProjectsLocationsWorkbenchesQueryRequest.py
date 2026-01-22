from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SeclmProjectsLocationsWorkbenchesQueryRequest(_messages.Message):
    """A SeclmProjectsLocationsWorkbenchesQueryRequest object.

  Fields:
    name: Required. Name of the resource
    workbenchQueryRequest: A WorkbenchQueryRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    workbenchQueryRequest = _messages.MessageField('WorkbenchQueryRequest', 2)