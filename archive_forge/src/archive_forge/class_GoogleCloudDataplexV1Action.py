from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Action(_messages.Message):
    """Action represents an issue requiring administrator action for
  resolution.

  Enums:
    CategoryValueValuesEnum: The category of issue associated with the action.

  Fields:
    asset: Output only. The relative resource name of the asset, of the form:
      projects/{project_number}/locations/{location_id}/lakes/{lake_id}/zones/
      {zone_id}/assets/{asset_id}.
    category: The category of issue associated with the action.
    dataLocations: The list of data locations associated with this action.
      Cloud Storage locations are represented as URI paths(E.g.
      gs://bucket/table1/year=2020/month=Jan/). BigQuery locations refer to
      resource names(E.g. bigquery.googleapis.com/projects/project-
      id/datasets/dataset-id).
    detectTime: The time that the issue was detected.
    failedSecurityPolicyApply: Details for issues related to applying security
      policy.
    incompatibleDataSchema: Details for issues related to incompatible schemas
      detected within data.
    invalidDataFormat: Details for issues related to invalid or unsupported
      data formats.
    invalidDataOrganization: Details for issues related to invalid data
      arrangement.
    invalidDataPartition: Details for issues related to invalid or unsupported
      data partition structure.
    issue: Detailed description of the issue requiring action.
    lake: Output only. The relative resource name of the lake, of the form:
      projects/{project_number}/locations/{location_id}/lakes/{lake_id}.
    missingData: Details for issues related to absence of data within managed
      resources.
    missingResource: Details for issues related to absence of a managed
      resource.
    name: Output only. The relative resource name of the action, of the form:
      projects/{project}/locations/{location}/lakes/{lake}/actions/{action} pr
      ojects/{project}/locations/{location}/lakes/{lake}/zones/{zone}/actions/
      {action} projects/{project}/locations/{location}/lakes/{lake}/zones/{zon
      e}/assets/{asset}/actions/{action}.
    unauthorizedResource: Details for issues related to lack of permissions to
      access data resources.
    zone: Output only. The relative resource name of the zone, of the form: pr
      ojects/{project_number}/locations/{location_id}/lakes/{lake_id}/zones/{z
      one_id}.
  """

    class CategoryValueValuesEnum(_messages.Enum):
        """The category of issue associated with the action.

    Values:
      CATEGORY_UNSPECIFIED: Unspecified category.
      RESOURCE_MANAGEMENT: Resource management related issues.
      SECURITY_POLICY: Security policy related issues.
      DATA_DISCOVERY: Data and discovery related issues.
    """
        CATEGORY_UNSPECIFIED = 0
        RESOURCE_MANAGEMENT = 1
        SECURITY_POLICY = 2
        DATA_DISCOVERY = 3
    asset = _messages.StringField(1)
    category = _messages.EnumField('CategoryValueValuesEnum', 2)
    dataLocations = _messages.StringField(3, repeated=True)
    detectTime = _messages.StringField(4)
    failedSecurityPolicyApply = _messages.MessageField('GoogleCloudDataplexV1ActionFailedSecurityPolicyApply', 5)
    incompatibleDataSchema = _messages.MessageField('GoogleCloudDataplexV1ActionIncompatibleDataSchema', 6)
    invalidDataFormat = _messages.MessageField('GoogleCloudDataplexV1ActionInvalidDataFormat', 7)
    invalidDataOrganization = _messages.MessageField('GoogleCloudDataplexV1ActionInvalidDataOrganization', 8)
    invalidDataPartition = _messages.MessageField('GoogleCloudDataplexV1ActionInvalidDataPartition', 9)
    issue = _messages.StringField(10)
    lake = _messages.StringField(11)
    missingData = _messages.MessageField('GoogleCloudDataplexV1ActionMissingData', 12)
    missingResource = _messages.MessageField('GoogleCloudDataplexV1ActionMissingResource', 13)
    name = _messages.StringField(14)
    unauthorizedResource = _messages.MessageField('GoogleCloudDataplexV1ActionUnauthorizedResource', 15)
    zone = _messages.StringField(16)