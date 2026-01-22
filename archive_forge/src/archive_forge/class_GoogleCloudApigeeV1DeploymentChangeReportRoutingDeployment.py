from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DeploymentChangeReportRoutingDeployment(_messages.Message):
    """Tuple representing a base path and the deployment containing it.

  Fields:
    apiProxy: Name of the deployed API proxy revision containing the base
      path.
    basepath: Base path receiving traffic.
    environment: Name of the environment in which the proxy is deployed.
    revision: Name of the deployed API proxy revision containing the base
      path.
  """
    apiProxy = _messages.StringField(1)
    basepath = _messages.StringField(2)
    environment = _messages.StringField(3)
    revision = _messages.StringField(4)