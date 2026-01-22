from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EndpointsApiService(_messages.Message):
    """Google Cloud Endpoints (https://cloud.google.com/endpoints)
  configuration. The Endpoints API Service provides tooling for serving Open
  API and gRPC endpoints via an NGINX proxy. Only valid for App Engine
  Flexible environment deployments.The fields here refer to the name and
  configuration ID of a "service" resource in the Service Management API
  (https://cloud.google.com/service-management/overview).

  Enums:
    RolloutStrategyValueValuesEnum: Endpoints rollout strategy. If FIXED,
      config_id must be specified. If MANAGED, config_id must be omitted.

  Fields:
    configId: Endpoints service configuration ID as specified by the Service
      Management API. For example "2016-09-19r1".By default, the rollout
      strategy for Endpoints is RolloutStrategy.FIXED. This means that
      Endpoints starts up with a particular configuration ID. When a new
      configuration is rolled out, Endpoints must be given the new
      configuration ID. The config_id field is used to give the configuration
      ID and is required in this case.Endpoints also has a rollout strategy
      called RolloutStrategy.MANAGED. When using this, Endpoints fetches the
      latest configuration and does not need the configuration ID. In this
      case, config_id must be omitted.
    disableTraceSampling: Enable or disable trace sampling. By default, this
      is set to false for enabled.
    name: Endpoints service name which is the name of the "service" resource
      in the Service Management API. For example
      "myapi.endpoints.myproject.cloud.goog"
    rolloutStrategy: Endpoints rollout strategy. If FIXED, config_id must be
      specified. If MANAGED, config_id must be omitted.
  """

    class RolloutStrategyValueValuesEnum(_messages.Enum):
        """Endpoints rollout strategy. If FIXED, config_id must be specified. If
    MANAGED, config_id must be omitted.

    Values:
      UNSPECIFIED_ROLLOUT_STRATEGY: Not specified. Defaults to FIXED.
      FIXED: Endpoints service configuration ID will be fixed to the
        configuration ID specified by config_id.
      MANAGED: Endpoints service configuration ID will be updated with each
        rollout.
    """
        UNSPECIFIED_ROLLOUT_STRATEGY = 0
        FIXED = 1
        MANAGED = 2
    configId = _messages.StringField(1)
    disableTraceSampling = _messages.BooleanField(2)
    name = _messages.StringField(3)
    rolloutStrategy = _messages.EnumField('RolloutStrategyValueValuesEnum', 4)