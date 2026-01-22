from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsGetRequest(_messages.Message):
    """A
  TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsGetRequest
  object.

  Enums:
    ViewValueValuesEnum: Optional. Defines the type of view of the deployment.
      When field is not present VIEW_BASIC is considered as default.

  Fields:
    name: Required. The name of the deployment. Case 1: If the name provided
      in the request is {deployment_id}@{revision_id}, then the revision with
      revision_id will be returned. Case 2: If the name provided in the
      request is {deployment}, then the current state of the deployment is
      returned.
    view: Optional. Defines the type of view of the deployment. When field is
      not present VIEW_BASIC is considered as default.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. Defines the type of view of the deployment. When field is
    not present VIEW_BASIC is considered as default.

    Values:
      DEPLOYMENT_VIEW_UNSPECIFIED: Unspecified enum value.
      DEPLOYMENT_VIEW_BASIC: View which only contains metadata.
      DEPLOYMENT_VIEW_FULL: View which contains metadata and files it
        encapsulates.
    """
        DEPLOYMENT_VIEW_UNSPECIFIED = 0
        DEPLOYMENT_VIEW_BASIC = 1
        DEPLOYMENT_VIEW_FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)