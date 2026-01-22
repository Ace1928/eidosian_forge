from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetBatchGetAssetsHistoryRequest(_messages.Message):
    """A CloudassetBatchGetAssetsHistoryRequest object.

  Enums:
    ContentTypeValueValuesEnum: Optional. The content type.

  Fields:
    assetNames: A list of the full names of the assets. See:
      https://cloud.google.com/asset-inventory/docs/resource-name-format
      Example: `//compute.googleapis.com/projects/my_project_123/zones/zone1/i
      nstances/instance1`. The request becomes a no-op if the asset name list
      is empty, and the max size of the asset name list is 100 in one request.
    contentType: Optional. The content type.
    parent: Required. The relative name of the root asset. It can only be an
      organization number (such as "organizations/123"), a project ID (such as
      "projects/my-project-id")", or a project number (such as
      "projects/12345").
    readTimeWindow_endTime: End time of the time window (inclusive). If not
      specified, the current timestamp is used instead.
    readTimeWindow_startTime: Start time of the time window (exclusive).
    relationshipTypes: Optional. A list of relationship types to output, for
      example: `INSTANCE_TO_INSTANCEGROUP`. This field should only be
      specified if content_type=RELATIONSHIP. * If specified: it outputs
      specified relationships' history on the [asset_names]. It returns an
      error if any of the [relationship_types] doesn't belong to the supported
      relationship types of the [asset_names] or if any of the [asset_names]'s
      types doesn't belong to the source types of the [relationship_types]. *
      Otherwise: it outputs the supported relationships' history on the
      [asset_names] or returns an error if any of the [asset_names]'s types
      has no relationship support. See [Introduction to Cloud Asset
      Inventory](https://cloud.google.com/asset-inventory/docs/overview) for
      all supported asset types and relationship types.
  """

    class ContentTypeValueValuesEnum(_messages.Enum):
        """Optional. The content type.

    Values:
      CONTENT_TYPE_UNSPECIFIED: Unspecified content type.
      RESOURCE: Resource metadata.
      IAM_POLICY: The actual IAM policy set on a resource.
      ORG_POLICY: The organization policy set on an asset.
      ACCESS_POLICY: The Access Context Manager policy set on an asset.
      OS_INVENTORY: The runtime OS Inventory information.
      RELATIONSHIP: The related resources.
    """
        CONTENT_TYPE_UNSPECIFIED = 0
        RESOURCE = 1
        IAM_POLICY = 2
        ORG_POLICY = 3
        ACCESS_POLICY = 4
        OS_INVENTORY = 5
        RELATIONSHIP = 6
    assetNames = _messages.StringField(1, repeated=True)
    contentType = _messages.EnumField('ContentTypeValueValuesEnum', 2)
    parent = _messages.StringField(3, required=True)
    readTimeWindow_endTime = _messages.StringField(4)
    readTimeWindow_startTime = _messages.StringField(5)
    relationshipTypes = _messages.StringField(6, repeated=True)