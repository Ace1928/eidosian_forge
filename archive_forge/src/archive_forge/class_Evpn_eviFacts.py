from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.evpn_evi.evpn_evi import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.evpn_evi import (
class Evpn_eviFacts(object):
    """The ios evpn_evi facts class"""

    def __init__(self, module, subspec='config', options='options'):
        self._module = module
        self.argument_spec = Evpn_eviArgs.argument_spec

    def get_evpn_evi_data(self, connection):
        return connection.get('show running-config | section ^l2vpn evpn instance .+$')

    def populate_facts(self, connection, ansible_facts, data=None):
        """Populate the facts for Evpn_evi network resource

        :param connection: the device connection
        :param ansible_facts: Facts dictionary
        :param data: previously collected conf

        :rtype: dictionary
        :returns: facts
        """
        facts = {}
        objs = []
        if not data:
            data = self.get_evpn_evi_data(connection)
        evpn_evi_parser = Evpn_eviTemplate(lines=data.splitlines(), module=self._module)
        objs = list(evpn_evi_parser.parse().values())
        ansible_facts['ansible_network_resources'].pop('evpn_evi', None)
        params = utils.remove_empties(evpn_evi_parser.validate_config(self.argument_spec, {'config': objs}, redact=True))
        facts['evpn_evi'] = params.get('config', [])
        ansible_facts['ansible_network_resources'].update(facts)
        return ansible_facts