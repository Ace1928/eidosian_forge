from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions as exception
class DictSerializer(ActionDispatcher):
    """Default request body serialization."""

    def serialize(self, data, action='default'):
        return self.dispatch(data, action=action)

    def default(self, data):
        return ''