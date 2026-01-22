from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoscalerStatusDetails(_messages.Message):
    """A AutoscalerStatusDetails object.

  Enums:
    TypeValueValuesEnum: The type of error, warning, or notice returned.
      Current set of possible values: - ALL_INSTANCES_UNHEALTHY (WARNING): All
      instances in the instance group are unhealthy (not in RUNNING state). -
      BACKEND_SERVICE_DOES_NOT_EXIST (ERROR): There is no backend service
      attached to the instance group. - CAPPED_AT_MAX_NUM_REPLICAS (WARNING):
      Autoscaler recommends a size greater than maxNumReplicas. -
      CUSTOM_METRIC_DATA_POINTS_TOO_SPARSE (WARNING): The custom metric
      samples are not exported often enough to be a credible base for
      autoscaling. - CUSTOM_METRIC_INVALID (ERROR): The custom metric that was
      specified does not exist or does not have the necessary labels. -
      MIN_EQUALS_MAX (WARNING): The minNumReplicas is equal to maxNumReplicas.
      This means the autoscaler cannot add or remove instances from the
      instance group. - MISSING_CUSTOM_METRIC_DATA_POINTS (WARNING): The
      autoscaler did not receive any data from the custom metric configured
      for autoscaling. - MISSING_LOAD_BALANCING_DATA_POINTS (WARNING): The
      autoscaler is configured to scale based on a load balancing signal but
      the instance group has not received any requests from the load balancer.
      - MODE_OFF (WARNING): Autoscaling is turned off. The number of instances
      in the group won't change automatically. The autoscaling configuration
      is preserved. - MODE_ONLY_UP (WARNING): Autoscaling is in the "Autoscale
      only out" mode. The autoscaler can add instances but not remove any. -
      MORE_THAN_ONE_BACKEND_SERVICE (ERROR): The instance group cannot be
      autoscaled because it has more than one backend service attached to it.
      - NOT_ENOUGH_QUOTA_AVAILABLE (ERROR): There is insufficient quota for
      the necessary resources, such as CPU or number of instances. -
      REGION_RESOURCE_STOCKOUT (ERROR): Shown only for regional autoscalers:
      there is a resource stockout in the chosen region. -
      SCALING_TARGET_DOES_NOT_EXIST (ERROR): The target to be scaled does not
      exist. - UNSUPPORTED_MAX_RATE_LOAD_BALANCING_CONFIGURATION (ERROR):
      Autoscaling does not work with an HTTP/S load balancer that has been
      configured for maxRate. - ZONE_RESOURCE_STOCKOUT (ERROR): For zonal
      autoscalers: there is a resource stockout in the chosen zone. For
      regional autoscalers: in at least one of the zones you're using there is
      a resource stockout. New values might be added in the future. Some of
      the values might not be available in all API versions.

  Fields:
    message: The status message.
    type: The type of error, warning, or notice returned. Current set of
      possible values: - ALL_INSTANCES_UNHEALTHY (WARNING): All instances in
      the instance group are unhealthy (not in RUNNING state). -
      BACKEND_SERVICE_DOES_NOT_EXIST (ERROR): There is no backend service
      attached to the instance group. - CAPPED_AT_MAX_NUM_REPLICAS (WARNING):
      Autoscaler recommends a size greater than maxNumReplicas. -
      CUSTOM_METRIC_DATA_POINTS_TOO_SPARSE (WARNING): The custom metric
      samples are not exported often enough to be a credible base for
      autoscaling. - CUSTOM_METRIC_INVALID (ERROR): The custom metric that was
      specified does not exist or does not have the necessary labels. -
      MIN_EQUALS_MAX (WARNING): The minNumReplicas is equal to maxNumReplicas.
      This means the autoscaler cannot add or remove instances from the
      instance group. - MISSING_CUSTOM_METRIC_DATA_POINTS (WARNING): The
      autoscaler did not receive any data from the custom metric configured
      for autoscaling. - MISSING_LOAD_BALANCING_DATA_POINTS (WARNING): The
      autoscaler is configured to scale based on a load balancing signal but
      the instance group has not received any requests from the load balancer.
      - MODE_OFF (WARNING): Autoscaling is turned off. The number of instances
      in the group won't change automatically. The autoscaling configuration
      is preserved. - MODE_ONLY_UP (WARNING): Autoscaling is in the "Autoscale
      only out" mode. The autoscaler can add instances but not remove any. -
      MORE_THAN_ONE_BACKEND_SERVICE (ERROR): The instance group cannot be
      autoscaled because it has more than one backend service attached to it.
      - NOT_ENOUGH_QUOTA_AVAILABLE (ERROR): There is insufficient quota for
      the necessary resources, such as CPU or number of instances. -
      REGION_RESOURCE_STOCKOUT (ERROR): Shown only for regional autoscalers:
      there is a resource stockout in the chosen region. -
      SCALING_TARGET_DOES_NOT_EXIST (ERROR): The target to be scaled does not
      exist. - UNSUPPORTED_MAX_RATE_LOAD_BALANCING_CONFIGURATION (ERROR):
      Autoscaling does not work with an HTTP/S load balancer that has been
      configured for maxRate. - ZONE_RESOURCE_STOCKOUT (ERROR): For zonal
      autoscalers: there is a resource stockout in the chosen zone. For
      regional autoscalers: in at least one of the zones you're using there is
      a resource stockout. New values might be added in the future. Some of
      the values might not be available in all API versions.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of error, warning, or notice returned. Current set of
    possible values: - ALL_INSTANCES_UNHEALTHY (WARNING): All instances in the
    instance group are unhealthy (not in RUNNING state). -
    BACKEND_SERVICE_DOES_NOT_EXIST (ERROR): There is no backend service
    attached to the instance group. - CAPPED_AT_MAX_NUM_REPLICAS (WARNING):
    Autoscaler recommends a size greater than maxNumReplicas. -
    CUSTOM_METRIC_DATA_POINTS_TOO_SPARSE (WARNING): The custom metric samples
    are not exported often enough to be a credible base for autoscaling. -
    CUSTOM_METRIC_INVALID (ERROR): The custom metric that was specified does
    not exist or does not have the necessary labels. - MIN_EQUALS_MAX
    (WARNING): The minNumReplicas is equal to maxNumReplicas. This means the
    autoscaler cannot add or remove instances from the instance group. -
    MISSING_CUSTOM_METRIC_DATA_POINTS (WARNING): The autoscaler did not
    receive any data from the custom metric configured for autoscaling. -
    MISSING_LOAD_BALANCING_DATA_POINTS (WARNING): The autoscaler is configured
    to scale based on a load balancing signal but the instance group has not
    received any requests from the load balancer. - MODE_OFF (WARNING):
    Autoscaling is turned off. The number of instances in the group won't
    change automatically. The autoscaling configuration is preserved. -
    MODE_ONLY_UP (WARNING): Autoscaling is in the "Autoscale only out" mode.
    The autoscaler can add instances but not remove any. -
    MORE_THAN_ONE_BACKEND_SERVICE (ERROR): The instance group cannot be
    autoscaled because it has more than one backend service attached to it. -
    NOT_ENOUGH_QUOTA_AVAILABLE (ERROR): There is insufficient quota for the
    necessary resources, such as CPU or number of instances. -
    REGION_RESOURCE_STOCKOUT (ERROR): Shown only for regional autoscalers:
    there is a resource stockout in the chosen region. -
    SCALING_TARGET_DOES_NOT_EXIST (ERROR): The target to be scaled does not
    exist. - UNSUPPORTED_MAX_RATE_LOAD_BALANCING_CONFIGURATION (ERROR):
    Autoscaling does not work with an HTTP/S load balancer that has been
    configured for maxRate. - ZONE_RESOURCE_STOCKOUT (ERROR): For zonal
    autoscalers: there is a resource stockout in the chosen zone. For regional
    autoscalers: in at least one of the zones you're using there is a resource
    stockout. New values might be added in the future. Some of the values
    might not be available in all API versions.

    Values:
      ALL_INSTANCES_UNHEALTHY: All instances in the instance group are
        unhealthy (not in RUNNING state).
      BACKEND_SERVICE_DOES_NOT_EXIST: There is no backend service attached to
        the instance group.
      CAPPED_AT_MAX_NUM_REPLICAS: Autoscaler recommends a size greater than
        maxNumReplicas.
      CUSTOM_METRIC_DATA_POINTS_TOO_SPARSE: The custom metric samples are not
        exported often enough to be a credible base for autoscaling.
      CUSTOM_METRIC_INVALID: The custom metric that was specified does not
        exist or does not have the necessary labels.
      MIN_EQUALS_MAX: The minNumReplicas is equal to maxNumReplicas. This
        means the autoscaler cannot add or remove instances from the instance
        group.
      MISSING_CUSTOM_METRIC_DATA_POINTS: The autoscaler did not receive any
        data from the custom metric configured for autoscaling.
      MISSING_LOAD_BALANCING_DATA_POINTS: The autoscaler is configured to
        scale based on a load balancing signal but the instance group has not
        received any requests from the load balancer.
      MODE_OFF: Autoscaling is turned off. The number of instances in the
        group won't change automatically. The autoscaling configuration is
        preserved.
      MODE_ONLY_SCALE_OUT: Autoscaling is in the "Autoscale only scale out"
        mode. Instances in the group will be only added.
      MODE_ONLY_UP: Autoscaling is in the "Autoscale only out" mode. Instances
        in the group will be only added.
      MORE_THAN_ONE_BACKEND_SERVICE: The instance group cannot be autoscaled
        because it has more than one backend service attached to it.
      NOT_ENOUGH_QUOTA_AVAILABLE: There is insufficient quota for the
        necessary resources, such as CPU or number of instances.
      REGION_RESOURCE_STOCKOUT: Showed only for regional autoscalers: there is
        a resource stockout in the chosen region.
      SCALING_TARGET_DOES_NOT_EXIST: The target to be scaled does not exist.
      SCHEDULED_INSTANCES_GREATER_THAN_AUTOSCALER_MAX: For some scaling
        schedules minRequiredReplicas is greater than maxNumReplicas.
        Autoscaler always recommends at most maxNumReplicas instances.
      SCHEDULED_INSTANCES_LESS_THAN_AUTOSCALER_MIN: For some scaling schedules
        minRequiredReplicas is less than minNumReplicas. Autoscaler always
        recommends at least minNumReplicas instances.
      UNKNOWN: <no description>
      UNSUPPORTED_MAX_RATE_LOAD_BALANCING_CONFIGURATION: Autoscaling does not
        work with an HTTP/S load balancer that has been configured for
        maxRate.
      ZONE_RESOURCE_STOCKOUT: For zonal autoscalers: there is a resource
        stockout in the chosen zone. For regional autoscalers: in at least one
        of the zones you're using there is a resource stockout.
    """
        ALL_INSTANCES_UNHEALTHY = 0
        BACKEND_SERVICE_DOES_NOT_EXIST = 1
        CAPPED_AT_MAX_NUM_REPLICAS = 2
        CUSTOM_METRIC_DATA_POINTS_TOO_SPARSE = 3
        CUSTOM_METRIC_INVALID = 4
        MIN_EQUALS_MAX = 5
        MISSING_CUSTOM_METRIC_DATA_POINTS = 6
        MISSING_LOAD_BALANCING_DATA_POINTS = 7
        MODE_OFF = 8
        MODE_ONLY_SCALE_OUT = 9
        MODE_ONLY_UP = 10
        MORE_THAN_ONE_BACKEND_SERVICE = 11
        NOT_ENOUGH_QUOTA_AVAILABLE = 12
        REGION_RESOURCE_STOCKOUT = 13
        SCALING_TARGET_DOES_NOT_EXIST = 14
        SCHEDULED_INSTANCES_GREATER_THAN_AUTOSCALER_MAX = 15
        SCHEDULED_INSTANCES_LESS_THAN_AUTOSCALER_MIN = 16
        UNKNOWN = 17
        UNSUPPORTED_MAX_RATE_LOAD_BALANCING_CONFIGURATION = 18
        ZONE_RESOURCE_STOCKOUT = 19
    message = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)