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
def _convert_dicts_to_entries(self, entries):
    """Convert dictionaries describing entries to Entry objects.

        The dictionaries come from the 'entries' field of the JSON
        dictionary you get when you GET a page of a collection. Each
        dictionary is the same as you'd get if you sent a GET request
        to the corresponding entry resource. So each of these
        dictionaries can be treated as a preprocessed representation
        of an entry resource, and turned into an Entry instance.

        :yield: A sequence of Entry instances.
        """
    for entry_dict in entries:
        resource_url = entry_dict['self_link']
        resource_type_link = entry_dict['resource_type_link']
        wadl_application = self._wadl_resource.application
        resource_type = wadl_application.get_resource_type(resource_type_link)
        resource = WadlResource(self._wadl_resource.application, resource_url, resource_type.tag)
        yield Resource._create_bound_resource(self._root, resource, entry_dict, self.JSON_MEDIA_TYPE, False)