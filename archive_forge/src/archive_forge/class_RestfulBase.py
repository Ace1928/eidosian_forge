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
class RestfulBase:
    """Base class for classes that know about lazr.restful services."""
    JSON_MEDIA_TYPE = 'application/json'

    def _transform_resources_to_links(self, dictionary):
        new_dictionary = {}
        for key, value in dictionary.items():
            if isinstance(value, Resource):
                value = value.self_link
            new_dictionary[self._get_external_param_name(key)] = value
        return new_dictionary

    def _get_external_param_name(self, param_name):
        """Turn a lazr.restful name into something to be sent over HTTP.

        For resources this may involve sticking '_link' or
        '_collection_link' on the end of the parameter name. For
        arguments to named operations, the parameter name is returned
        as is.
        """
        return param_name