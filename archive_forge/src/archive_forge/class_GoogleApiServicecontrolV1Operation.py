from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1Operation(_messages.Message):
    """Represents information regarding an operation.

  Enums:
    ImportanceValueValuesEnum: DO NOT USE. This is an experimental field.

  Messages:
    LabelsValue: Labels describing the operation. Only the following labels
      are allowed: - Labels describing monitored resources as defined in the
      service configuration. - Default labels of metric values. When
      specified, labels defined in the metric value override these default. -
      The following labels defined by Google Cloud Platform: -
      `cloud.googleapis.com/location` describing the location where the
      operation happened, - `servicecontrol.googleapis.com/user_agent`
      describing the user agent of the API request, -
      `servicecontrol.googleapis.com/service_agent` describing the service
      used to handle the API request (e.g. ESP), -
      `servicecontrol.googleapis.com/platform` describing the platform where
      the API is served, such as App Engine, Compute Engine, or Kubernetes
      Engine.
    UserLabelsValue: Private Preview. This feature is only available for
      approved services. User defined labels for the resource that this
      operation is associated with.

  Fields:
    consumerId: Identity of the consumer who is using the service. This field
      should be filled in for the operations initiated by a consumer, but not
      for service-initiated operations that are not related to a specific
      consumer. - This can be in one of the following formats: -
      project:PROJECT_ID, - project`_`number:PROJECT_NUMBER, -
      projects/PROJECT_ID or PROJECT_NUMBER, - folders/FOLDER_NUMBER, -
      organizations/ORGANIZATION_NUMBER, - api`_`key:API_KEY.
    endTime: End time of the operation. Required when the operation is used in
      ServiceController.Report, but optional when the operation is used in
      ServiceController.Check.
    importance: DO NOT USE. This is an experimental field.
    labels: Labels describing the operation. Only the following labels are
      allowed: - Labels describing monitored resources as defined in the
      service configuration. - Default labels of metric values. When
      specified, labels defined in the metric value override these default. -
      The following labels defined by Google Cloud Platform: -
      `cloud.googleapis.com/location` describing the location where the
      operation happened, - `servicecontrol.googleapis.com/user_agent`
      describing the user agent of the API request, -
      `servicecontrol.googleapis.com/service_agent` describing the service
      used to handle the API request (e.g. ESP), -
      `servicecontrol.googleapis.com/platform` describing the platform where
      the API is served, such as App Engine, Compute Engine, or Kubernetes
      Engine.
    logEntries: Represents information to be logged.
    metricValueSets: Represents information about this operation. Each
      MetricValueSet corresponds to a metric defined in the service
      configuration. The data type used in the MetricValueSet must agree with
      the data type specified in the metric definition. Within a single
      operation, it is not allowed to have more than one MetricValue instances
      that have the same metric names and identical label value combinations.
      If a request has such duplicated MetricValue instances, the entire
      request is rejected with an invalid argument error.
    operationId: Identity of the operation. This must be unique within the
      scope of the service that generated the operation. If the service calls
      Check() and Report() on the same operation, the two calls should carry
      the same id. UUID version 4 is recommended, though not required. In
      scenarios where an operation is computed from existing information and
      an idempotent id is desirable for deduplication purpose, UUID version 5
      is recommended. See RFC 4122 for details.
    operationName: Fully qualified name of the operation. Reserved for future
      use.
    quotaProperties: Represents the properties needed for quota check.
      Applicable only if this operation is for a quota check request. If this
      is not specified, no quota check will be performed.
    resources: The resources that are involved in the operation. The maximum
      supported number of entries in this field is 100.
    startTime: Required. Start time of the operation.
    traceSpans: Unimplemented. A list of Cloud Trace spans. The span names
      shall contain the id of the destination project which can be either the
      produce or the consumer project.
    userLabels: Private Preview. This feature is only available for approved
      services. User defined labels for the resource that this operation is
      associated with.
  """

    class ImportanceValueValuesEnum(_messages.Enum):
        """DO NOT USE. This is an experimental field.

    Values:
      LOW: Allows data caching, batching, and aggregation. It provides higher
        performance with higher data loss risk.
      HIGH: Disables data aggregation to minimize data loss. It is for
        operations that contains significant monetary value or audit trail.
        This feature only applies to the client libraries.
      DEBUG: Deprecated. Do not use. Disables data aggregation and enables
        additional validation logic. It should only be used during the
        onboarding process. It is only available to Google internal services,
        and the service must be approved by chemist-dev@google.com in order to
        use this level.
      PROMOTED: Used internally by Chemist.
    """
        LOW = 0
        HIGH = 1
        DEBUG = 2
        PROMOTED = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels describing the operation. Only the following labels are
    allowed: - Labels describing monitored resources as defined in the service
    configuration. - Default labels of metric values. When specified, labels
    defined in the metric value override these default. - The following labels
    defined by Google Cloud Platform: - `cloud.googleapis.com/location`
    describing the location where the operation happened, -
    `servicecontrol.googleapis.com/user_agent` describing the user agent of
    the API request, - `servicecontrol.googleapis.com/service_agent`
    describing the service used to handle the API request (e.g. ESP), -
    `servicecontrol.googleapis.com/platform` describing the platform where the
    API is served, such as App Engine, Compute Engine, or Kubernetes Engine.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UserLabelsValue(_messages.Message):
        """Private Preview. This feature is only available for approved services.
    User defined labels for the resource that this operation is associated
    with.

    Messages:
      AdditionalProperty: An additional property for a UserLabelsValue object.

    Fields:
      additionalProperties: Additional properties of type UserLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UserLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    consumerId = _messages.StringField(1)
    endTime = _messages.StringField(2)
    importance = _messages.EnumField('ImportanceValueValuesEnum', 3)
    labels = _messages.MessageField('LabelsValue', 4)
    logEntries = _messages.MessageField('GoogleApiServicecontrolV1LogEntry', 5, repeated=True)
    metricValueSets = _messages.MessageField('GoogleApiServicecontrolV1MetricValueSet', 6, repeated=True)
    operationId = _messages.StringField(7)
    operationName = _messages.StringField(8)
    quotaProperties = _messages.MessageField('GoogleApiServicecontrolV1QuotaProperties', 9)
    resources = _messages.MessageField('GoogleApiServicecontrolV1ResourceInfo', 10, repeated=True)
    startTime = _messages.StringField(11)
    traceSpans = _messages.MessageField('GoogleApiServicecontrolV1TraceSpan', 12, repeated=True)
    userLabels = _messages.MessageField('UserLabelsValue', 13)