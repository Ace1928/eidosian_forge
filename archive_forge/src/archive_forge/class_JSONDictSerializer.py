from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions as exception
class JSONDictSerializer(DictSerializer):
    """Default JSON request body serialization."""

    def default(self, data):

        def sanitizer(obj):
            return str(obj)
        return jsonutils.dumps(data, default=sanitizer)