from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetAssetsListRequest(_messages.Message):
    """A CloudassetAssetsListRequest object.

  Enums:
    ContentTypeValueValuesEnum: Asset content type. If not specified, no
      content but the asset name will be returned.

  Fields:
    assetTypes: A list of asset types to take a snapshot for. For example:
      "compute.googleapis.com/Disk". Regular expression is also supported. For
      example: * "compute.googleapis.com.*" snapshots resources whose asset
      type starts with "compute.googleapis.com". * ".*Instance" snapshots
      resources whose asset type ends with "Instance". * ".*Instance.*"
      snapshots resources whose asset type contains "Instance". See
      [RE2](https://github.com/google/re2/wiki/Syntax) for all supported
      regular expression syntax. If the regular expression does not match any
      supported asset type, an INVALID_ARGUMENT error will be returned. If
      specified, only matching assets will be returned, otherwise, it will
      snapshot all asset types. See [Introduction to Cloud Asset
      Inventory](https://cloud.google.com/asset-inventory/docs/overview) for
      all supported asset types.
    contentType: Asset content type. If not specified, no content but the
      asset name will be returned.
    pageSize: The maximum number of assets to be returned in a single
      response. Default is 100, minimum is 1, and maximum is 1000.
    pageToken: The `next_page_token` returned from the previous
      `ListAssetsResponse`, or unspecified for the first `ListAssetsRequest`.
      It is a continuation of a prior `ListAssets` call, and the API should
      return the next page of assets.
    parent: Required. Name of the organization, folder, or project the assets
      belong to. Format: "organizations/[organization-number]" (such as
      "organizations/123"), "projects/[project-id]" (such as "projects/my-
      project-id"), "projects/[project-number]" (such as "projects/12345"), or
      "folders/[folder-number]" (such as "folders/12345").
    readTime: Timestamp to take an asset snapshot. This can only be set to a
      timestamp between the current time and the current time minus 35 days
      (inclusive). If not specified, the current time will be used. Due to
      delays in resource data collection and indexing, there is a volatile
      window during which running the same query may get different results.
    relationshipTypes: A list of relationship types to output, for example:
      `INSTANCE_TO_INSTANCEGROUP`. This field should only be specified if
      content_type=RELATIONSHIP. * If specified: it snapshots specified
      relationships. It returns an error if any of the [relationship_types]
      doesn't belong to the supported relationship types of the [asset_types]
      or if any of the [asset_types] doesn't belong to the source types of the
      [relationship_types]. * Otherwise: it snapshots the supported
      relationships for all [asset_types] or returns an error if any of the
      [asset_types] has no relationship support. An unspecified asset types
      field means all supported asset_types. See [Introduction to Cloud Asset
      Inventory](https://cloud.google.com/asset-inventory/docs/overview) for
      all supported asset types and relationship types.
  """

    class ContentTypeValueValuesEnum(_messages.Enum):
        """Asset content type. If not specified, no content but the asset name
    will be returned.

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
    assetTypes = _messages.StringField(1, repeated=True)
    contentType = _messages.EnumField('ContentTypeValueValuesEnum', 2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    readTime = _messages.StringField(6)
    relationshipTypes = _messages.StringField(7, repeated=True)