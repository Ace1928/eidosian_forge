from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def GetAttachmentPointFlag():
    return base.Argument('--attachment-point', required=True, help='Resource to which the policy is attached. For valid formats, see https://cloud.google.com/iam/help/deny/attachment-point.')