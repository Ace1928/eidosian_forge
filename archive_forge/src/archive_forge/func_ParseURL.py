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
def ParseURL(self, url):
    """Parse a URL into a Resource.

    This method does not yet handle "api.google.com" in place of
    "www.googleapis.com/api/version".

    Searches self.parsers_by_url to find a _ResourceParser. The parsers_by_url
    attribute is a deeply nested dictionary, where each key corresponds to
    a URL segment. The first segment is an API's base URL (eg.
    "https://www.googleapis.com/compute/v1/"), and after that it's each
    remaining token in the URL, split on '/'. Then a path down the tree is
    followed, keyed by the extracted pieces of the provided URL. If the key in
    the tree is a literal string, like "project" in .../project/{project}/...,
    the token from the URL must match exactly. If it's a parameter, like
    "{project}", then any token can match it, and that token is stored in a
    dict of params to with the associated key ("project" in this case). If there
    are no URL tokens left, and one of the keys at the current level is None,
    the None points to a _ResourceParser that can turn the collected
    params into a Resource.

    Args:
      url: str, The URL of the resource.

    Returns:
      Resource, The resource indicated by the provided URL.

    Raises:
      InvalidResourceException: If the provided URL could not be turned into
          a cloud resource.
    """
    match = _URL_RE.match(url)
    if not match:
        raise InvalidResourceException(url, reason='unknown API host')
    api_name, api_version, resource_path = resource_util.SplitEndpointUrl(url)
    try:
        versions = apis_internal._GetVersions(api_name)
    except apis_util.UnknownAPIError:
        raise InvalidResourceException(url, 'unknown api {}'.format(api_name))
    if api_version not in versions:
        if HasOverriddenEndpoint(api_name):
            api_version = self.registered_apis.get(api_name, apis_internal._GetDefaultVersion(api_name))
    if api_version not in versions:
        raise InvalidResourceException(url, 'unknown api version {}'.format(api_version))
    tokens = [api_name, api_version] + resource_path.split('/')
    endpoint = url[:-len(resource_path)]
    try:
        self.RegisterApiByName(api_name, api_version=api_version)
    except apis_util.UnknownAPIError:
        raise InvalidResourceException(url, 'unknown api {}'.format(api_name))
    except apis_util.UnknownVersionError:
        raise InvalidResourceException(url, 'unknown api version {}'.format(api_version))
    params = []
    cur_level = self.parsers_by_url
    for i, token in enumerate(tokens):
        if token in cur_level:
            cur_level = cur_level[token]
            continue
        param, next_level = ('', {})
        for param, next_level in six.iteritems(cur_level):
            if param == '{}':
                break
        else:
            raise InvalidResourceException(url, reason='Could not parse at [{}]'.format(token))
        if len(next_level) == 1 and None in next_level:
            token = '/'.join(tokens[i:])
            params.append(urllib.parse.unquote(token))
            cur_level = next_level
            break
        params.append(urllib.parse.unquote(token))
        cur_level = next_level
    if None not in cur_level:
        raise InvalidResourceException(url, 'Url too short.')
    subcollection, parser = cur_level[None]
    params = dict(zip(parser.collection_info.GetParams(subcollection), params))
    return parser.ParseResourceId(None, params, base_url=endpoint, subcollection=subcollection)