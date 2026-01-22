from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
from typing import Optional
from absl import flags
import googleapiclient
from googleapiclient import http as http_request
from googleapiclient import model
import httplib2
import bq_utils
from clients import utils as bq_client_utils
class BigqueryModel(model.JsonModel):
    """Adds optional global parameters to all requests."""

    def __init__(self, trace=None, quota_project_id: Optional[str]=None, **kwds):
        super().__init__(**kwds)
        self.trace = trace
        self.quota_project_id = quota_project_id

    def request(self, headers, path_params, query_params, body_value):
        """Updates outgoing request."""
        if 'trace' not in query_params and self.trace:
            headers['cookie'] = self.trace
        if self.quota_project_id:
            headers['x-goog-user-project'] = self.quota_project_id
        return super().request(headers, path_params, query_params, body_value)

    def response(self, resp, content):
        """Convert the response wire format into a Python object."""
        logging.info('Response from server with status code: %s', resp['status'])
        return super().response(resp, content)