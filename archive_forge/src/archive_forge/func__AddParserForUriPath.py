from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
import uritemplate
def _AddParserForUriPath(self, api_name, api_version, subcollection, parser, path):
    """Registers parser for given path."""
    tokens = [api_name, api_version] + path.split('/')
    cur_level = self.parsers_by_url
    while tokens:
        token = tokens.pop(0)
        if token[0] == '{' and token[-1] == '}':
            token = '{}'
        if token not in cur_level:
            cur_level[token] = {}
        cur_level = cur_level[token]
    if None in cur_level:
        raise AmbiguousResourcePath(cur_level[None], parser.collection_info.name)
    cur_level[None] = (subcollection, parser)