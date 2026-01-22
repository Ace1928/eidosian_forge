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
def _transform_resources_to_links(self, dictionary):
    new_dictionary = {}
    for key, value in dictionary.items():
        if isinstance(value, Resource):
            value = value.self_link
        new_dictionary[self._get_external_param_name(key)] = value
    return new_dictionary