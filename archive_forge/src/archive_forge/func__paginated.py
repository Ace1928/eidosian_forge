import abc
import contextlib
import hashlib
import os
from urllib import parse
from troveclient.apiclient import base
from troveclient.apiclient import exceptions
from troveclient import common
from troveclient import utils
def _paginated(self, url, response_key, limit=None, marker=None, query_strings=None):
    query_strings = query_strings or {}
    url = common.append_query_strings(url, limit=limit, marker=marker, **query_strings)
    resp, body = self.api.client.get(url)
    if not body:
        raise Exception('Call to ' + url + ' did not return a body.')
    links = body.get('links', [])
    next_links = [link['href'] for link in links if link['rel'] == 'next']
    next_marker = None
    for link in next_links:
        parsed_url = parse.urlparse(link)
        query_dict = dict(parse.parse_qsl(parsed_url.query))
        next_marker = query_dict.get('marker')
    data = [self.resource_class(self, res) for res in body[response_key]]
    return common.Paginated(data, next_marker=next_marker, links=links)