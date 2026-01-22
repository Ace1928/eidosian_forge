from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Deployment(_messages.Message):
    """A GoogleCloudApigeeV1Deployment object.

  Enums:
    ProxyDeploymentTypeValueValuesEnum: Output only. The type of the
      deployment (standard or extensible) Deployed proxy revision will be
      marked as extensible in following 2 cases. 1. The deployed proxy
      revision uses extensible policies. 2. If a environment supports
      flowhooks and flow hook is configured.
    StateValueValuesEnum: Current state of the deployment. **Note**: This
      field is displayed only when viewing deployment status.

  Fields:
    apiProxy: API proxy.
    deployStartTime: Time the API proxy was marked `deployed` in the control
      plane in millisconds since epoch.
    environment: Environment.
    errors: Errors reported for this deployment. Populated only when state ==
      ERROR. **Note**: This field is displayed only when viewing deployment
      status.
    instances: Status reported by each runtime instance. **Note**: This field
      is displayed only when viewing deployment status.
    pods: Status reported by runtime pods. **Note**: **This field is
      deprecated**. Runtime versions 1.3 and above report instance level
      status rather than pod status.
    proxyDeploymentType: Output only. The type of the deployment (standard or
      extensible) Deployed proxy revision will be marked as extensible in
      following 2 cases. 1. The deployed proxy revision uses extensible
      policies. 2. If a environment supports flowhooks and flow hook is
      configured.
    revision: API proxy revision.
    routeConflicts: Conflicts in the desired state routing configuration. The
      presence of conflicts does not cause the state to be `ERROR`, but it
      will mean that some of the deployment's base paths are not routed to its
      environment. If the conflicts change, the state will transition to
      `PROGRESSING` until the latest configuration is rolled out to all
      instances. **Note**: This field is displayed only when viewing
      deployment status.
    serviceAccount: The full resource name of Cloud IAM Service Account that
      this deployment is using, eg, `projects/-/serviceAccounts/{email}`.
    state: Current state of the deployment. **Note**: This field is displayed
      only when viewing deployment status.
  """

    class ProxyDeploymentTypeValueValuesEnum(_messages.Enum):
        """Output only. The type of the deployment (standard or extensible)
    Deployed proxy revision will be marked as extensible in following 2 cases.
    1. The deployed proxy revision uses extensible policies. 2. If a
    environment supports flowhooks and flow hook is configured.

    Values:
      PROXY_DEPLOYMENT_TYPE_UNSPECIFIED: Default value till public preview.
        After public preview this value should not be returned.
      STANDARD: Deployment will be of type Standard if only Standard proxies
        are used
      EXTENSIBLE: Proxy will be of type Extensible if deployments uses one or
        more Extensible proxies
    """
        PROXY_DEPLOYMENT_TYPE_UNSPECIFIED = 0
        STANDARD = 1
        EXTENSIBLE = 2

    class StateValueValuesEnum(_messages.Enum):
        """Current state of the deployment. **Note**: This field is displayed
    only when viewing deployment status.

    Values:
      RUNTIME_STATE_UNSPECIFIED: This value should never be returned.
      READY: Runtime has loaded the deployment.
      PROGRESSING: Deployment is not fully ready in the runtime.
      ERROR: Encountered an error with the deployment that requires
        intervention.
    """
        RUNTIME_STATE_UNSPECIFIED = 0
        READY = 1
        PROGRESSING = 2
        ERROR = 3
    apiProxy = _messages.StringField(1)
    deployStartTime = _messages.IntegerField(2)
    environment = _messages.StringField(3)
    errors = _messages.MessageField('GoogleRpcStatus', 4, repeated=True)
    instances = _messages.MessageField('GoogleCloudApigeeV1InstanceDeploymentStatus', 5, repeated=True)
    pods = _messages.MessageField('GoogleCloudApigeeV1PodStatus', 6, repeated=True)
    proxyDeploymentType = _messages.EnumField('ProxyDeploymentTypeValueValuesEnum', 7)
    revision = _messages.StringField(8)
    routeConflicts = _messages.MessageField('GoogleCloudApigeeV1DeploymentChangeReportRoutingConflict', 9, repeated=True)
    serviceAccount = _messages.StringField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)