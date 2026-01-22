from __future__ import absolute_import
import six
import json
import logging
import platform
from six.moves.urllib.parse import urlencode
from googleapiclient.errors import HttpError
class MediaModel(JsonModel):
    """Model class for requests that return Media.

  Serializes and de-serializes between JSON and the Python
  object representation of HTTP request, and returns the raw bytes
  of the response body.
  """
    accept = '*/*'
    content_type = 'application/json'
    alt_param = 'media'

    def deserialize(self, content):
        return content

    @property
    def no_content_response(self):
        return ''