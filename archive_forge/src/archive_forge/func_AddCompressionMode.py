from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddCompressionMode(parser):
    """Add support for --compression-mode flag."""
    return parser.add_argument('--compression-mode', choices=['DISABLED', 'AUTOMATIC'], type=arg_utils.ChoiceToEnumName, help="      Compress text responses using Brotli or gzip compression, based on\n      the client's Accept-Encoding header. Two modes are supported:\n      AUTOMATIC (recommended) - automatically uses the best compression based\n      on the Accept-Encoding header sent by the client. In most cases, this\n      will result in Brotli compression being favored.\n      DISABLED - disables compression. Existing compressed responses cached\n      by Cloud CDN will not be served to clients.\n      ")