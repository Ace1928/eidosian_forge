from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DotnetSettings(_messages.Message):
    """Settings for Dotnet client libraries.

  Messages:
    RenamedResourcesValue: Map from full resource types to the effective short
      name for the resource. This is used when otherwise resource named from
      different services would cause naming collisions. Example entry:
      "datalabeling.googleapis.com/Dataset": "DataLabelingDataset"
    RenamedServicesValue: Map from original service names to renamed versions.
      This is used when the default generated types would cause a naming
      conflict. (Neither name is fully-qualified.) Example: Subscriber to
      SubscriberServiceApi.

  Fields:
    common: Some settings.
    forcedNamespaceAliases: Namespaces which must be aliased in snippets due
      to a known (but non-generator-predictable) naming collision
    handwrittenSignatures: Method signatures (in the form
      "service.method(signature)") which are provided separately, so shouldn't
      be generated. Snippets *calling* these methods are still generated,
      however.
    ignoredResources: List of full resource types to ignore during generation.
      This is typically used for API-specific Location resources, which should
      be handled by the generator as if they were actually the common Location
      resources. Example entry: "documentai.googleapis.com/Location"
    renamedResources: Map from full resource types to the effective short name
      for the resource. This is used when otherwise resource named from
      different services would cause naming collisions. Example entry:
      "datalabeling.googleapis.com/Dataset": "DataLabelingDataset"
    renamedServices: Map from original service names to renamed versions. This
      is used when the default generated types would cause a naming conflict.
      (Neither name is fully-qualified.) Example: Subscriber to
      SubscriberServiceApi.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class RenamedResourcesValue(_messages.Message):
        """Map from full resource types to the effective short name for the
    resource. This is used when otherwise resource named from different
    services would cause naming collisions. Example entry:
    "datalabeling.googleapis.com/Dataset": "DataLabelingDataset"

    Messages:
      AdditionalProperty: An additional property for a RenamedResourcesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        RenamedResourcesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a RenamedResourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class RenamedServicesValue(_messages.Message):
        """Map from original service names to renamed versions. This is used when
    the default generated types would cause a naming conflict. (Neither name
    is fully-qualified.) Example: Subscriber to SubscriberServiceApi.

    Messages:
      AdditionalProperty: An additional property for a RenamedServicesValue
        object.

    Fields:
      additionalProperties: Additional properties of type RenamedServicesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a RenamedServicesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    common = _messages.MessageField('CommonLanguageSettings', 1)
    forcedNamespaceAliases = _messages.StringField(2, repeated=True)
    handwrittenSignatures = _messages.StringField(3, repeated=True)
    ignoredResources = _messages.StringField(4, repeated=True)
    renamedResources = _messages.MessageField('RenamedResourcesValue', 5)
    renamedServices = _messages.MessageField('RenamedServicesValue', 6)