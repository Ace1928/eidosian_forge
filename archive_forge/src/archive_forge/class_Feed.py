from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Feed(_messages.Message):
    """An asset feed used to export asset updates to a destinations. An asset
  feed filter controls what updates are exported. The asset feed must be
  created within a project, organization, or folder. Supported destinations
  are: Pub/Sub topics.

  Enums:
    ContentTypeValueValuesEnum: Asset content type. If not specified, no
      content but the asset name and type will be returned.

  Fields:
    assetNames: A list of the full names of the assets to receive updates. You
      must specify either or both of asset_names and asset_types. Only asset
      updates matching specified asset_names or asset_types are exported to
      the feed. Example: `//compute.googleapis.com/projects/my_project_123/zon
      es/zone1/instances/instance1`. For a list of the full names for
      supported asset types, see [Resource name format](/asset-
      inventory/docs/resource-name-format).
    assetTypes: A list of types of the assets to receive updates. You must
      specify either or both of asset_names and asset_types. Only asset
      updates matching specified asset_names or asset_types are exported to
      the feed. Example: `"compute.googleapis.com/Disk"` For a list of all
      supported asset types, see [Supported asset types](/asset-
      inventory/docs/supported-asset-types).
    condition: A condition which determines whether an asset update should be
      published. If specified, an asset will be returned only when the
      expression evaluates to true. When set, `expression` field in the `Expr`
      must be a valid [CEL expression] (https://github.com/google/cel-spec) on
      a TemporalAsset with name `temporal_asset`. Example: a Feed with
      expression ("temporal_asset.deleted == true") will only publish Asset
      deletions. Other fields of `Expr` are optional. See our [user
      guide](https://cloud.google.com/asset-inventory/docs/monitoring-asset-
      changes-with-condition) for detailed instructions.
    contentType: Asset content type. If not specified, no content but the
      asset name and type will be returned.
    feedOutputConfig: Required. Feed output configuration defining where the
      asset updates are published to.
    name: Required. The format will be
      projects/{project_number}/feeds/{client-assigned_feed_identifier} or
      folders/{folder_number}/feeds/{client-assigned_feed_identifier} or
      organizations/{organization_number}/feeds/{client-
      assigned_feed_identifier} The client-assigned feed identifier must be
      unique within the parent project/folder/organization.
    relationshipTypes: A list of relationship types to output, for example:
      `INSTANCE_TO_INSTANCEGROUP`. This field should only be specified if
      content_type=RELATIONSHIP. * If specified: it outputs specified
      relationship updates on the [asset_names] or the [asset_types]. It
      returns an error if any of the [relationship_types] doesn't belong to
      the supported relationship types of the [asset_names] or [asset_types],
      or any of the [asset_names] or the [asset_types] doesn't belong to the
      source types of the [relationship_types]. * Otherwise: it outputs the
      supported relationships of the types of [asset_names] and [asset_types]
      or returns an error if any of the [asset_names] or the [asset_types] has
      no replationship support. See [Introduction to Cloud Asset
      Inventory](https://cloud.google.com/asset-inventory/docs/overview) for
      all supported asset types and relationship types.
  """

    class ContentTypeValueValuesEnum(_messages.Enum):
        """Asset content type. If not specified, no content but the asset name
    and type will be returned.

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
    assetTypes = _messages.StringField(2, repeated=True)
    condition = _messages.MessageField('Expr', 3)
    contentType = _messages.EnumField('ContentTypeValueValuesEnum', 4)
    feedOutputConfig = _messages.MessageField('FeedOutputConfig', 5)
    name = _messages.StringField(6)
    relationshipTypes = _messages.StringField(7, repeated=True)