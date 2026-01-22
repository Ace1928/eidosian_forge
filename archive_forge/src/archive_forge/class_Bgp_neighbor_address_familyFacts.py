from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.bgp_neighbor_address_family.bgp_neighbor_address_family import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.bgp_neighbor_address_family import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
class Bgp_neighbor_address_familyFacts(object):
    """The iosxr bgp_neighbor_address_family facts class"""

    def __init__(self, module, subspec='config', options='options'):
        self._module = module
        self.argument_spec = Bgp_neighbor_address_familyArgs.argument_spec
        spec = deepcopy(self.argument_spec)
        if subspec:
            if options:
                facts_argument_spec = spec[subspec][options]
            else:
                facts_argument_spec = spec[subspec]
        else:
            facts_argument_spec = spec
        self.generated_spec = utils.generate_dict(facts_argument_spec)

    def get_config(self, connection):
        return connection.get('show running-config router bgp')

    def populate_facts(self, connection, ansible_facts, data=None):
        """Populate the facts for Bgp_address_family network resource
        :param connection: the device connection
        :param ansible_facts: Facts dictionary
        :param data: previously collected conf
        :rtype: dictionary
        :returns: facts
        """
        facts = {}
        objs = []
        if not data:
            data = self.get_config(connection)
        nbr_data = flatten_config(data, 'neighbor')
        data = flatten_config(nbr_data, 'vrf')
        bgp_global_parser = Bgp_neighbor_address_familyTemplate(lines=data.splitlines())
        objs = bgp_global_parser.parse()
        if objs:
            top_lvl_nbrs = objs.get('vrfs', {}).pop('vrf_', {})
            objs['neighbors'] = self._post_parse(top_lvl_nbrs).get('neighbors', [])
            if 'vrfs' in objs:
                for vrf in objs['vrfs'].values():
                    vrf['neighbors'] = self._post_parse(vrf)['neighbors']
                objs['vrfs'] = list(objs['vrfs'].values())
        ansible_facts['ansible_network_resources'].pop('bgp_neighbor_address_family', None)
        params = utils.remove_empties(utils.validate_config(self.argument_spec, {'config': objs}))
        facts['bgp_neighbor_address_family'] = params.get('config', {})
        ansible_facts['ansible_network_resources'].update(facts)
        return ansible_facts

    def _post_parse(self, data):
        """Converts the intermediate data structure
            to valid format as per argspec.
        :param obj: dict
        """
        if 'neighbors' in data:
            data['neighbors'] = sorted(list(data['neighbors'].values()), key=lambda k, s='neighbor_address': k[s])
            for nbr in data['neighbors']:
                nbr['address_family'] = list(nbr['address_family'].values())
        return data