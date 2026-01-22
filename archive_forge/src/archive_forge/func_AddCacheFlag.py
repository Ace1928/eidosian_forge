from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.calliope import walker
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core import resources
from googlecloudsdk.core.cache import exceptions as cache_exceptions
from googlecloudsdk.core.cache import file_cache
from googlecloudsdk.core.cache import resource_cache
import six
def AddCacheFlag(parser):
    """Adds the persistent cache flag to the parser."""
    parser.add_argument('--cache', metavar='CACHE_NAME', default=_CACHE_RI_DEFAULT, help='The cache name to operate on. May be prefixed by "{}" for resource cache names. If only the prefix is specified then the default cache name for that prefix is used.'.format(_CACHE_RI_DEFAULT))