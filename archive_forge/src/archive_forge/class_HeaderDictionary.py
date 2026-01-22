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
class HeaderDictionary:
    """A dictionary that bridges httplib2's and wadllib's expectations.

    httplib2 expects all header dictionary access to give lowercase
    header names. wadllib expects to access the header exactly as it's
    specified in the WADL file, which means the official HTTP header name.

    This class transforms keys to lowercase before doing a lookup on
    the underlying dictionary. That way wadllib can pass in the
    official header name and httplib2 will get the lowercased name.
    """

    def __init__(self, wrapped_dictionary):
        self.wrapped_dictionary = wrapped_dictionary

    def get(self, key, default=None):
        """Retrieve a value, converting the key to lowercase."""
        return self.wrapped_dictionary.get(key.lower())

    def __getitem__(self, key):
        """Retrieve a value, converting the key to lowercase."""
        value = self.get(key, missing)
        if value is missing:
            raise KeyError(key)
        return value