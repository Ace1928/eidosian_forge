from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CrawledUrl(_messages.Message):
    """A CrawledUrl resource represents a URL that was crawled during a
  ScanRun. Web Security Scanner Service crawls the web applications, following
  all links within the scope of sites, to find the URLs to test against.

  Fields:
    body: The body of the request that was used to visit the URL.
    httpMethod: The http method of the request that was used to visit the URL,
      in uppercase.
    url: The URL that was crawled.
  """
    body = _messages.StringField(1)
    httpMethod = _messages.StringField(2)
    url = _messages.StringField(3)