from oslo_serialization import jsonutils
from keystone import exception
from keystone.i18n import _
class JsonHomeResources(object):
    """JSON Home resource data."""
    __resources = {}
    __serialized_resource_data = None

    @classmethod
    def _reset(cls):
        cls.__resources.clear()
        cls.__serialized_resource_data = None

    @classmethod
    def append_resource(cls, rel, data):
        cls.__resources[rel] = data
        cls.__serialized_resource_data = None

    @classmethod
    def resources(cls):
        if cls.__serialized_resource_data is None:
            cls.__serialized_resource_data = jsonutils.dumps(cls.__resources)
        return {'resources': jsonutils.loads(cls.__serialized_resource_data)}