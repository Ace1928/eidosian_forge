from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
class IamData(object):
    """A holder object for IAM related information specified in yaml command."""

    def __init__(self, data):
        self.message_type_overrides = data.get('message_type_overrides', {})
        self.set_iam_policy_request_path = data.get('set_iam_policy_request_path')
        self.enable_condition = data.get('enable_condition', False)
        self.hide_special_member_types = data.get('hide_special_member_types', False)
        self.policy_version = data.get('policy_version', None)
        self.get_iam_policy_version_path = data.get('get_iam_policy_version_path', 'options.requestedPolicyVersion')