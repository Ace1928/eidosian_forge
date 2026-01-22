from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisVersionsSpecsRollbackRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisVersionsSpecsRollbackRequest
  object.

  Fields:
    name: Required. The spec being rolled back.
    rollbackApiSpecRequest: A RollbackApiSpecRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    rollbackApiSpecRequest = _messages.MessageField('RollbackApiSpecRequest', 2)