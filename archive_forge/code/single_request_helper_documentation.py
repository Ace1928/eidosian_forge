from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.compute import operation_quota_utils
from googlecloudsdk.api_lib.compute import utils
import six
Makes single request.

  Args:
    service: a BaseApiService Object.
    method: a string of method name.
    request_body: a protocol buffer requesting the requests.

  Returns:
    a length-one response list and error list.
  