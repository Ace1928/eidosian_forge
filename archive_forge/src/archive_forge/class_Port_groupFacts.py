from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.port_group.port_group import Port_groupArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
class Port_groupFacts(object):
    """ The sonic port group fact class
    """

    def __init__(self, module, subspec='config', options='options'):
        self._module = module
        self.argument_spec = Port_groupArgs.argument_spec
        spec = deepcopy(self.argument_spec)
        if subspec:
            if options:
                facts_argument_spec = spec[subspec][options]
            else:
                facts_argument_spec = spec[subspec]
        else:
            facts_argument_spec = spec
        self.generated_spec = utils.generate_dict(facts_argument_spec)

    def populate_facts(self, connection, ansible_facts, data=None):
        """ Populate the facts for port groups
        :param connection: the device connection
        :param ansible_facts: Facts dictionary
        :param data: previously collected conf
        :rtype: dictionary
        :returns: facts
        """
        if not data:
            data = self.get_port_groups()
        objs = []
        for conf in data:
            if conf:
                obj = self.render_config(self.generated_spec, conf)
                if obj:
                    objs.append(obj)
        ansible_facts['ansible_network_resources'].pop('port_group', None)
        facts = {}
        if objs:
            facts['port_group'] = []
            params = utils.validate_config(self.argument_spec, {'config': objs})
            if params:
                facts['port_group'].extend(params['config'])
        ansible_facts['ansible_network_resources'].update(facts)
        return ansible_facts

    def render_config(self, spec, conf):
        """
        Render config as dictionary structure and delete keys
          from spec for null values

        :param spec: The facts tree, generated from the argspec
        :param conf: The configuration
        :rtype: dictionary
        :returns: The generated config
        """
        return conf

    def get_port_groups(self):
        """Get all the port group configurations"""
        pgs_request = [{'path': 'data/openconfig-port-group:port-groups/port-group', 'method': GET}]
        try:
            pgs_response = edit_config(self._module, to_request(self._module, pgs_request))
        except ConnectionError as exc:
            self._module.fail_json(msg=str(exc), code=exc.code)
        pgs_config = []
        if 'openconfig-port-group:port-group' in pgs_response[0][1]:
            pgs_config = pgs_response[0][1].get('openconfig-port-group:port-group', [])
        pgs = []
        for pg_config in pgs_config:
            pg = dict()
            if 'config' in pg_config:
                pg['id'] = pg_config['id']
                speed_str = pg_config['config'].get('speed', None)
                if speed_str:
                    pg['speed'] = speed_str.split(':', 1)[-1]
                    pgs.append(pg)
        return pgs