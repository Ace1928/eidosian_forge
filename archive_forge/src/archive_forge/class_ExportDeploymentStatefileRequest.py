from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportDeploymentStatefileRequest(_messages.Message):
    """A request to export a state file passed to a 'ExportDeploymentStatefile'
  call.

  Fields:
    draft: Optional. If this flag is set to true, the exported deployment
      state file will be the draft state. This will enable the draft file to
      be validated before copying it over to the working state on unlock.
  """
    draft = _messages.BooleanField(1)