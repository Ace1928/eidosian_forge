from __future__ import absolute_import, division, print_function
import platform
import re
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
class Neighbors(LegacyFactsBase):
    COMMANDS = ['show lldp neighbors', 'show lldp neighbors detail']

    def populate(self):
        super(Neighbors, self).populate()
        all_neighbors = self.responses[0]
        if 'LLDP not configured' not in all_neighbors:
            neighbors = self.parse(self.responses[1])
            self.facts['neighbors'] = self.parse_neighbors(neighbors)

    def parse(self, data):
        parsed = list()
        values = None
        for line in data.split('\n'):
            if not line:
                continue
            if line[0] == ' ':
                values += '\n%s' % line
            elif line.startswith('Interface'):
                if values:
                    parsed.append(values)
                values = line
        if values:
            parsed.append(values)
        return parsed

    def parse_neighbors(self, data):
        facts = dict()
        for item in data:
            interface = self.parse_interface(item)
            host = self.parse_host(item)
            port = self.parse_port(item)
            if interface not in facts:
                facts[interface] = list()
            facts[interface].append(dict(host=host, port=port))
        return facts

    def parse_interface(self, data):
        match = re.search('^Interface:\\s+(\\S+),', data)
        return match.group(1)

    def parse_host(self, data):
        match = re.search('SysName:\\s+(.+)$', data, re.M)
        if match:
            return match.group(1)

    def parse_port(self, data):
        match = re.search('PortDescr:\\s+(.+)$', data, re.M)
        if match:
            return match.group(1)