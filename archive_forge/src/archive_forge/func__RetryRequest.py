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
def _RetryRequest(http, num_retries, req_type, sleep, rand, uri, method, *args, **kwargs):
    """Conditionally retries an HTTP request.

  This is a wrapper around http_request._retry_request. If the original request
  fails with a specific permission error, retry it once without the
  x-goog-user-project header.

  Args:
    http: Http object to be used to execute request.
    num_retries: Maximum number of retries.
    req_type: Type of the request (used for logging retries).
    sleep: Function to sleep for random time between retries.
    rand: Function to sleep for random time between retries.
    uri: URI to be requested.
    method: HTTP method to be used.
    *args: Additional arguments passed to http.request.
    **kwargs: Additional arguments passed to http.request.

  Returns:
    resp, content - Response from the http request (may be HTTP 5xx).
  """
    resp, content = _ORIGINAL_GOOGLEAPI_CLIENT_RETRY_REQUEST(http, num_retries, req_type, sleep, rand, uri, method, *args, **kwargs)
    if int(resp.status) == 403:
        data = json.loads(content.decode('utf-8'))
        if isinstance(data, dict) and 'message' in data['error']:
            err_message = data['error']['message']
            if 'roles/serviceusage.serviceUsageConsumer' in err_message:
                if 'headers' in kwargs and 'x-goog-user-project' in kwargs['headers']:
                    del kwargs['headers']['x-goog-user-project']
                    logging.info('Retrying request without the x-goog-user-project header')
                    resp, content = _ORIGINAL_GOOGLEAPI_CLIENT_RETRY_REQUEST(http, num_retries, req_type, sleep, rand, uri, method, *args, **kwargs)
    return (resp, content)