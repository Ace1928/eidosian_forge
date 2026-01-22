from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DeploymentChangeReportRoutingConflict(_messages.Message):
    """Describes a routing conflict that may cause a deployment not to receive
  traffic at some base path.

  Fields:
    conflictingDeployment: Existing base path/deployment causing the conflict.
    description: Human-readable description of this conflict.
    environmentGroup: Name of the environment group in which this conflict
      exists.
  """
    conflictingDeployment = _messages.MessageField('GoogleCloudApigeeV1DeploymentChangeReportRoutingDeployment', 1)
    description = _messages.StringField(2)
    environmentGroup = _messages.StringField(3)