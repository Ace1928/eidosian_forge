from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDatacollectorsPatchRequest(_messages.Message):
    """A ApigeeOrganizationsDatacollectorsPatchRequest object.

  Fields:
    googleCloudApigeeV1DataCollector: A GoogleCloudApigeeV1DataCollector
      resource to be passed as the request body.
    name: Required. Name of the data collector in the following format:
      `organizations/{org}/datacollectors/{data_collector_id}`.
    updateMask: List of fields to be updated.
  """
    googleCloudApigeeV1DataCollector = _messages.MessageField('GoogleCloudApigeeV1DataCollector', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)