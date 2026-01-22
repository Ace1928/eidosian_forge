from __future__ import absolute_import
from apitools.base.py import base_api
from samples.fusiontables_sample.fusiontables_v1 import fusiontables_v1_messages as messages
def SqlGet(self, request, global_params=None, download=None):
    """Executes an SQL SELECT/SHOW/DESCRIBE statement.

      Args:
        request: (FusiontablesQuerySqlGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        download: (Download, default: None) If present, download
            data from the request via this stream.
      Returns:
        (Sqlresponse) The response message.
      """
    config = self.GetMethodConfig('SqlGet')
    return self._RunMethod(config, request, global_params=global_params, download=download)