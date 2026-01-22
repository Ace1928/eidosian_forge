from email.message import Message
from io import BytesIO
from json import dumps, loads
import sys
from wadllib.application import Resource as WadlResource
from lazr.restfulclient import __version__
from lazr.restfulclient._browser import Browser, RestfulHttp
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError
from lazr.uri import URI
def _handle_201_response(self, url, response, content):
    """Handle the creation of a new resource by fetching it."""
    wadl_response = self.wadl_method.response.bind(HeaderDictionary(response))
    wadl_parameter = wadl_response.get_parameter('Location')
    wadl_resource = wadl_parameter.linked_resource
    response, content = self.root._browser._request(wadl_resource.url)
    return Resource._create_bound_resource(self.root, wadl_resource, content, response['content-type'])