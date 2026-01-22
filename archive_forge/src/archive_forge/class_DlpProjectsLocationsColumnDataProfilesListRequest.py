from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsColumnDataProfilesListRequest(_messages.Message):
    """A DlpProjectsLocationsColumnDataProfilesListRequest object.

  Fields:
    filter: Allows filtering. Supported syntax: * Filter expressions are made
      up of one or more restrictions. * Restrictions can be combined by `AND`
      or `OR` logical operators. A sequence of restrictions implicitly uses
      `AND`. * A restriction has the form of `{field} {operator} {value}`. *
      Supported fields/values: - `table_data_profile_name` - The name of the
      related table data profile. - `project_id` - The Google Cloud project
      ID. (REQUIRED) - `dataset_id` - The BigQuery dataset ID. (REQUIRED) -
      `table_id` - The BigQuery table ID. (REQUIRED) - `field_id` - The ID of
      the BigQuery field. - `info_type` - The infotype detected in the
      resource. - `sensitivity_level` - HIGH|MEDIUM|LOW - `data_risk_level`:
      How much risk is associated with this data. - `status_code` - an RPC
      status code as defined in https://github.com/googleapis/googleapis/blob/
      master/google/rpc/code.proto * The operator must be `=` for project_id,
      dataset_id, and table_id. Other filters also support `!=`. Examples: *
      project_id = 12345 AND status_code = 1 * project_id = 12345 AND
      sensitivity_level = HIGH * project_id = 12345 AND info_type =
      STREET_ADDRESS The length of this field should be no more than 500
      characters.
    orderBy: Comma separated list of fields to order by, followed by `asc` or
      `desc` postfix. This list is case insensitive. The default sorting order
      is ascending. Redundant space characters are insignificant. Only one
      order field at a time is allowed. Examples: * `project_id asc` *
      `table_id` * `sensitivity_level desc` Supported fields are: -
      `project_id`: The Google Cloud project ID. - `dataset_id`: The ID of a
      BigQuery dataset. - `table_id`: The ID of a BigQuery table. -
      `sensitivity_level`: How sensitive the data in a column is, at most. -
      `data_risk_level`: How much risk is associated with this data. -
      `profile_last_generated`: When the profile was last updated in epoch
      seconds.
    pageSize: Size of the page. This value can be limited by the server. If
      zero, server returns a page of max size 100.
    pageToken: Page token to continue retrieval.
    parent: Required. Resource name of the organization or project, for
      example `organizations/433245324/locations/europe` or `projects/project-
      id/locations/asia`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)