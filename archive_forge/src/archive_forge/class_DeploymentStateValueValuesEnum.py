from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentStateValueValuesEnum(_messages.Enum):
    """The state of the Operator's deployment

    Values:
      DEPLOYMENT_STATE_UNSPECIFIED: Deployment's state cannot be determined
      NOT_INSTALLED: Deployment is not installed
      INSTALLED: Deployment is installed
      ERROR: Deployment was attempted to be installed, but has errors
      PENDING: Deployment is installing or terminating
    """
    DEPLOYMENT_STATE_UNSPECIFIED = 0
    NOT_INSTALLED = 1
    INSTALLED = 2
    ERROR = 3
    PENDING = 4