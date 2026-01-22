from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def GetPolicyFileFlag():
    return base.Argument('--policy-file', required=True, help='Path to the file that contains the policy, in JSON or YAML format. For valid syntax, see https://cloud.google.com/iam/help/deny/policy-syntax.')