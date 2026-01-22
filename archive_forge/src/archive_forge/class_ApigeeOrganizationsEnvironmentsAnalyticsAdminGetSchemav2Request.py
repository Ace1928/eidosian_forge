from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsAnalyticsAdminGetSchemav2Request(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsAnalyticsAdminGetSchemav2Request
  object.

  Fields:
    disableCache: Flag that specifies whether the schema is be read from the
      database or cache. Set to `true` to read the schema from the database.
      Defaults to cache.
    name: Required. Path to the schema. Use the following structure in your
      request:
      `organizations/{org}/environments/{env}/analytics/admin/schemav2`.
    type: Required. Name of the dataset for which you want to retrieve the
      schema. For example: `fact` or `agg_cus1`
  """
    disableCache = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    type = _messages.StringField(3)