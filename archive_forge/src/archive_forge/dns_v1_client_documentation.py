from __future__ import absolute_import
from apitools.base.py import base_api
from samples.dns_sample.dns_v1 import dns_v1_messages as messages
Enumerate ResourceRecordSets that have been created but not yet deleted.

      Args:
        request: (DnsResourceRecordSetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResourceRecordSetsListResponse) The response message.
      