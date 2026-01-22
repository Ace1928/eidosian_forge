from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from six.moves.urllib import parse
def _AppendLocation(endpoint, location):
    scheme, netloc, path, params, query, fragment = parse.urlparse(endpoint)
    netloc = '{}-{}'.format(location, netloc)
    return parse.urlunparse((scheme, netloc, path, params, query, fragment))