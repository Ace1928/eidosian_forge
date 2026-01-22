from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionTemplatesCreateRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionTemplatesCreateRequest object.

  Fields:
    parent: Required. The parent resource where this session template will be
      created.
    sessionTemplate: A SessionTemplate resource to be passed as the request
      body.
  """
    parent = _messages.StringField(1, required=True)
    sessionTemplate = _messages.MessageField('SessionTemplate', 2)