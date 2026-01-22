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
@classmethod
def _create_bound_resource(cls, root, resource, representation=None, representation_media_type='application/json', representation_needs_processing=True, representation_definition=None, param_name=None):
    """Create a lazr.restful Resource subclass from a wadllib Resource.

        :param resource: The wadllib Resource to wrap.
        :param representation: A previously fetched representation of
            this resource, to be reused. If not provided, this method
            will act just like the Resource constructor.
        :param representation_media_type: The media type of any previously
            fetched representation.
        :param representation_needs_processing: Set to False if the
            'representation' parameter should be used as
            is.
        :param representation_definition: A wadllib
            RepresentationDefinition object describing the structure
            of this representation. Used in cases when the representation
            isn't the result of sending a standard GET to the resource.
        :param param_name: The name of the link that was followed to get
            to this resource.
        :return: An instance of the appropriate lazr.restful Resource
            subclass.
        """
    type_url = resource.type_url
    resource_type = urlparse(type_url)[-1]
    default = Entry
    if type_url.endswith('-page') or (param_name is not None and param_name.endswith('_collection_link')):
        default = Collection
    r_class = root.RESOURCE_TYPE_CLASSES.get(resource_type, default)
    if representation is not None:
        resource = resource.bind(representation, representation_media_type, representation_needs_processing, representation_definition=representation_definition)
    else:
        pass
    return r_class(root, resource)