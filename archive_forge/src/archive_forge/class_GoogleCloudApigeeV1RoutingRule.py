from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1RoutingRule(_messages.Message):
    """A GoogleCloudApigeeV1RoutingRule object.

  Fields:
    basepath: URI path prefix used to route to the specified environment. May
      contain one or more wildcards. For example, path segments consisting of
      a single `*` character will match any string.
    deploymentGroup: Name of a deployment group in an environment bound to the
      environment group in the following format:
      `organizations/{org}/environment/{env}/deploymentGroups/{group}` Only
      one of environment or deployment_group will be set.
    envGroupRevision: The env group config revision_id when this rule was
      added or last updated. This value is set when the rule is created and
      will only update if the the environment_id changes. It is used to
      determine if the runtime is up to date with respect to this rule. This
      field is omitted from the IngressConfig unless the
      GetDeployedIngressConfig API is called with view=FULL.
    environment: Name of an environment bound to the environment group in the
      following format: `organizations/{org}/environments/{env}`. Only one of
      environment or deployment_group will be set.
    otherTargets: Conflicting targets, which will be resource names specifying
      either deployment groups or environments.
    receiver: The resource name of the proxy revision that is receiving this
      basepath in the following format:
      `organizations/{org}/apis/{api}/revisions/{rev}`. This field is omitted
      from the IngressConfig unless the GetDeployedIngressConfig API is called
      with view=FULL.
    updateTime: The unix timestamp when this rule was updated. This is updated
      whenever env_group_revision is updated. This field is omitted from the
      IngressConfig unless the GetDeployedIngressConfig API is called with
      view=FULL.
  """
    basepath = _messages.StringField(1)
    deploymentGroup = _messages.StringField(2)
    envGroupRevision = _messages.IntegerField(3)
    environment = _messages.StringField(4)
    otherTargets = _messages.StringField(5, repeated=True)
    receiver = _messages.StringField(6)
    updateTime = _messages.StringField(7)