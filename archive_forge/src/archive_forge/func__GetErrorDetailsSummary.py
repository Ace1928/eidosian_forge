from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
import six
def _GetErrorDetailsSummary(error_info):
    """Returns a string summarizing `error_info`.

  Attempts to interpret error_info as an error JSON returned by the Apigee
  management API. If successful, the returned string will be an error message
  from that data structure - either its top-level error message, or a list of
  precondition violations.

  If `error_info` can't be parsed, or has no known error message, returns a YAML
  formatted copy of `error_info` instead.

  Args:
    error_info: a dictionary containing the error data structure returned by the
      Apigee Management API.
  """
    try:
        if 'details' in error_info:
            violations = []
            for item in error_info['details']:
                detail_types = ('type.googleapis.com/google.rpc.QuotaFailure', 'type.googleapis.com/google.rpc.PreconditionFailure', 'type.googleapis.com/edge.configstore.bundle.BadBundle')
                if item['@type'] in detail_types and 'violations' in item:
                    violations += item['violations']
            descriptions = [violation['description'] for violation in violations]
            if descriptions:
                return error_info['message'] + '\n' + yaml.dump(descriptions)
        return error_info['message']
    except KeyError:
        return '\n' + yaml.dump(error_info)