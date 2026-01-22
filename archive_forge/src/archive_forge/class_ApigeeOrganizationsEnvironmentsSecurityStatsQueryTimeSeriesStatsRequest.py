from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityStatsQueryTimeSeriesStatsRequest(_messages.Message):
    """A
  ApigeeOrganizationsEnvironmentsSecurityStatsQueryTimeSeriesStatsRequest
  object.

  Fields:
    googleCloudApigeeV1QueryTimeSeriesStatsRequest: A
      GoogleCloudApigeeV1QueryTimeSeriesStatsRequest resource to be passed as
      the request body.
    orgenv: Required. Should be of the form organizations//environments/.
  """
    googleCloudApigeeV1QueryTimeSeriesStatsRequest = _messages.MessageField('GoogleCloudApigeeV1QueryTimeSeriesStatsRequest', 1)
    orgenv = _messages.StringField(2, required=True)