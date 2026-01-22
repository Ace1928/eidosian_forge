from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _get_listener(client, listener_id_or_name):
    return neutronV20.find_resource_by_name_or_id(client, 'listener', listener_id_or_name)