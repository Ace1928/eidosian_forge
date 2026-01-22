from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddProxySecretArn(parser, required=False):
    parser.add_argument('--proxy-secret-arn', required=required, help='ARN of the AWS Secrets Manager secret that contains a proxy configuration.')