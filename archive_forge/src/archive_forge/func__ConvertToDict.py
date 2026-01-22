from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.services import common_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _ConvertToDict(policy):
    """ConvertToDict command.

  Args:
    policy: consumerPolicy to be convert to orderedDict.

  Returns:
    orderedDict.
  """
    output = {'name': policy.name, 'enableRules': [], 'updateTime': policy.updateTime, 'createTime': policy.createTime, 'etag': policy.etag}
    for enable_rule in policy.enableRules:
        if enable_rule.services:
            output['enableRules'].append({'services': list(enable_rule.services)})
    if not policy.enableRules:
        del output['enableRules']
    if policy.updateTime == _INVALID_TIMESTAMP:
        del output['updateTime']
    if policy.createTime == _INVALID_TIMESTAMP:
        del output['createTime']
    return output