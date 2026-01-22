from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1p7beta1Resource(_messages.Message):
    """A representation of a Google Cloud resource.

  Messages:
    DataValue: The content of the resource, in which some sensitive fields are
      removed and may not be present.

  Fields:
    data: The content of the resource, in which some sensitive fields are
      removed and may not be present.
    discoveryDocumentUri: The URL of the discovery document containing the
      resource's JSON schema. Example:
      `https://www.googleapis.com/discovery/v1/apis/compute/v1/rest` This
      value is unspecified for resources that do not have an API based on a
      discovery document, such as Cloud Bigtable.
    discoveryName: The JSON schema name listed in the discovery document.
      Example: `Project` This value is unspecified for resources that do not
      have an API based on a discovery document, such as Cloud Bigtable.
    location: The location of the resource in Google Cloud, such as its zone
      and region. For more information, see
      https://cloud.google.com/about/locations/.
    parent: The full name of the immediate parent of this resource. See
      [Resource Names](https://cloud.google.com/apis/design/resource_names#ful
      l_resource_name) for more information. For Google Cloud assets, this
      value is the parent resource defined in the [IAM policy
      hierarchy](https://cloud.google.com/iam/docs/overview#policy_hierarchy).
      Example: `//cloudresourcemanager.googleapis.com/projects/my_project_123`
      For third-party assets, this field may be set differently.
    resourceUrl: The REST URL for accessing the resource. An HTTP `GET`
      request using this URL returns the resource itself. Example:
      `https://cloudresourcemanager.googleapis.com/v1/projects/my-project-123`
      This value is unspecified for resources without a REST API.
    version: The API version. Example: `v1`
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DataValue(_messages.Message):
        """The content of the resource, in which some sensitive fields are
    removed and may not be present.

    Messages:
      AdditionalProperty: An additional property for a DataValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    data = _messages.MessageField('DataValue', 1)
    discoveryDocumentUri = _messages.StringField(2)
    discoveryName = _messages.StringField(3)
    location = _messages.StringField(4)
    parent = _messages.StringField(5)
    resourceUrl = _messages.StringField(6)
    version = _messages.StringField(7)