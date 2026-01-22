from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.iam import completers as iam_completers
from googlecloudsdk.command_lib.util.args import labels_util
def AddBackendAuthServiceAccountFlag(parser):
    """Adds the backend auth service account flag."""
    parser.add_argument('--backend-auth-service-account', help='      Service account which will be used to sign tokens for backends with       authentication configured.\n      ')