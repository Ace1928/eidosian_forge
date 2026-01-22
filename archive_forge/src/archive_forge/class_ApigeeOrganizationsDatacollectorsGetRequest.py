from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDatacollectorsGetRequest(_messages.Message):
    """A ApigeeOrganizationsDatacollectorsGetRequest object.

  Fields:
    name: Required. Name of the data collector in the following format:
      `organizations/{org}/datacollectors/{data_collector_id}`.
  """
    name = _messages.StringField(1, required=True)