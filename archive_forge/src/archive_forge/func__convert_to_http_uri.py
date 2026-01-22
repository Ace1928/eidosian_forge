import io
import logging
import urllib.parse
from smart_open import utils, constants
import http.client as httplib
def _convert_to_http_uri(webhdfs_url):
    """
    Convert webhdfs uri to http url and return it as text

    Parameters
    ----------
    webhdfs_url: str
        A URL starting with webhdfs://
    """
    split_uri = urllib.parse.urlsplit(webhdfs_url)
    netloc = split_uri.hostname
    if split_uri.port:
        netloc += ':{}'.format(split_uri.port)
    query = split_uri.query
    if split_uri.username:
        query += ('&' if query else '') + 'user.name=' + urllib.parse.quote(split_uri.username)
    return urllib.parse.urlunsplit(('http', netloc, '/webhdfs/v1' + split_uri.path, query, ''))