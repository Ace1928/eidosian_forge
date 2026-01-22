from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1Form(_messages.Message):
    """A form is a data model that groups related parameters that can be
  collected from the user. The process in which the agent prompts the user and
  collects parameter values from the user is called form filling. A form can
  be added to a page. When form filling is done, the filled parameters will be
  written to the session.

  Fields:
    parameters: Parameters to collect from the user.
  """
    parameters = _messages.MessageField('GoogleCloudDialogflowCxV3beta1FormParameter', 1, repeated=True)