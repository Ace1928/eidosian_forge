from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Service(_messages.Message):
    """`Service` is the root object of the configuration schema. It describes
  basic information like the name of the service and the exposed API
  interfaces, and delegates other aspects to configuration sub-sections.
  Example:      type: google.api.Service     config_version: 1     name:
  calendar.googleapis.com     title: Google Calendar API     apis:     - name:
  google.calendar.Calendar     backend:       rules:       - selector: "*"
  address: calendar.example.com

  Fields:
    apis: A list of API interfaces exported by this service. Only the `name`
      field of the google.protobuf.Api needs to be provided by the
      configuration author, as the remaining fields will be derived from the
      IDL during the normalization process. It is an error to specify an API
      interface here which cannot be resolved against the associated IDL
      files.
    authentication: Auth configuration.
    backend: API backend configuration.
    billing: Billing configuration of the service.
    configVersion: The version of the service configuration. The config
      version may influence interpretation of the configuration, for example,
      to determine defaults. This is documented together with applicable
      options. The current default for the config version itself is `3`.
    context: Context configuration.
    control: Configuration for the service control plane.
    customError: Custom error configuration.
    documentation: Additional API documentation.
    enums: A list of all enum types included in this API service.  Enums
      referenced directly or indirectly by the `apis` are automatically
      included.  Enums which are not referenced but shall be included should
      be listed here by name. Example:      enums:     - name:
      google.someapi.v1.SomeEnum
    http: HTTP configuration.
    id: A unique ID for a specific instance of this message, typically
      assigned by the client for tracking purpose. If empty, the server may
      choose to generate one instead.
    logging: Logging configuration of the service.
    logs: Defines the logs used by this service.
    metrics: Defines the metrics used by this service.
    monitoredResources: Defines the monitored resources used by this service.
      This is required by the Service.monitoring and Service.logging
      configurations.
    monitoring: Monitoring configuration of the service.
    name: The DNS address at which this service is available, e.g.
      `calendar.googleapis.com`.
    producerProjectId: The id of the Google developer project that owns the
      service. Members of this project can manage the service configuration,
      manage consumption of the service, etc.
    projectProperties: Configuration of per-consumer project properties.
    quota: Quota configuration.
    systemParameters: Configuration for system parameters.
    systemTypes: A list of all proto message types included in this API
      service. It serves similar purpose as [google.api.Service.types], except
      that these types are not needed by user-defined APIs. Therefore, they
      will not show up in the generated discovery doc. This field should only
      be used to define system APIs in ESF.
    title: The product title associated with this service.
    types: A list of all proto message types included in this API service.
      Types referenced directly or indirectly by the `apis` are automatically
      included.  Messages which are not referenced but shall be included, such
      as types used by the `google.protobuf.Any` type, should be listed here
      by name. Example:      types:     - name: google.protobuf.Int32
    usage: Configuration controlling usage of this service.
    visibility: API visibility configuration.
  """
    apis = _messages.MessageField('Api', 1, repeated=True)
    authentication = _messages.MessageField('Authentication', 2)
    backend = _messages.MessageField('Backend', 3)
    billing = _messages.MessageField('Billing', 4)
    configVersion = _messages.IntegerField(5, variant=_messages.Variant.UINT32)
    context = _messages.MessageField('Context', 6)
    control = _messages.MessageField('Control', 7)
    customError = _messages.MessageField('CustomError', 8)
    documentation = _messages.MessageField('Documentation', 9)
    enums = _messages.MessageField('Enum', 10, repeated=True)
    http = _messages.MessageField('Http', 11)
    id = _messages.StringField(12)
    logging = _messages.MessageField('Logging', 13)
    logs = _messages.MessageField('LogDescriptor', 14, repeated=True)
    metrics = _messages.MessageField('MetricDescriptor', 15, repeated=True)
    monitoredResources = _messages.MessageField('MonitoredResourceDescriptor', 16, repeated=True)
    monitoring = _messages.MessageField('Monitoring', 17)
    name = _messages.StringField(18)
    producerProjectId = _messages.StringField(19)
    projectProperties = _messages.MessageField('ProjectProperties', 20)
    quota = _messages.MessageField('Quota', 21)
    systemParameters = _messages.MessageField('SystemParameters', 22)
    systemTypes = _messages.MessageField('Type', 23, repeated=True)
    title = _messages.StringField(24)
    types = _messages.MessageField('Type', 25, repeated=True)
    usage = _messages.MessageField('Usage', 26)
    visibility = _messages.MessageField('Visibility', 27)