from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.hostname.hostname import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.hostname import (
class HostnameFacts(object):
    """The vyos hostname facts class"""

    def __init__(self, module, subspec='config', options='options'):
        self._module = module
        self.argument_spec = HostnameArgs.argument_spec

    def get_config(self, connection):
        return connection.get('show configuration commands | grep host-name')

    def populate_facts(self, connection, ansible_facts, data=None):
        """Populate the facts for Snmp_server network resource

        :param connection: the device connection
        :param ansible_facts: Facts dictionary
        :param data: previously collected conf

        :rtype: dictionary
        :returns: facts
        """
        facts = {}
        objs = []
        config_lines = []
        if not data:
            data = self.get_config(connection)
        for resource in data.splitlines():
            config_lines.append(re.sub("'", '', resource))
        hostname_parser = HostnameTemplate(lines=config_lines, module=self._module)
        objs = hostname_parser.parse()
        ansible_facts['ansible_network_resources'].pop('hostname', None)
        params = utils.remove_empties(hostname_parser.validate_config(self.argument_spec, {'config': objs}, redact=True))
        facts['hostname'] = params.get('config', {})
        ansible_facts['ansible_network_resources'].update(facts)
        return ansible_facts