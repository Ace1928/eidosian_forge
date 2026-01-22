from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.facts.facts import Facts
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.rm_templates.ogs import (
def _service_object_compare(self, want, have):
    service_obj = 'service_object'
    services_obj = 'services_object'
    port_obj = 'port_object'
    for name, entry in iteritems(want):
        h_item = have.pop(name, {})
        if entry != h_item and name != 'object_type':
            if h_item and entry.get('group_object'):
                self.addcmd(entry, 'og_name', False)
                self._add_group_object_cmd(entry, h_item)
                continue
            if h_item:
                self._add_object_cmd(entry, h_item, service_obj, ['protocol'])
            else:
                protocol = entry.get('protocol')
                if protocol:
                    entry['name'] = '{0} {1}'.format(name, protocol)
                self.addcmd(entry, 'og_name', False)
                self.compare(['description'], entry, h_item)
            if entry.get('group_object'):
                self._add_group_object_cmd(entry, h_item)
                continue
            if entry.get(service_obj):
                if entry[service_obj].get('protocol'):
                    self._compare_object_diff(entry, h_item, service_obj, 'protocol', ['service_object'], service_obj)
            elif entry.get(services_obj):
                if h_item:
                    h_item = self.convert_list_to_dict(val=h_item, source='source_port', destination='destination_port')
                entry = self.convert_list_to_dict(val=entry, source='source_port', destination='destination_port')
                command_len = len(self.commands)
                for k, v in iteritems(entry):
                    if h_item:
                        h_service_item = h_item.pop(k, {})
                        if h_service_item != v:
                            self.compare([services_obj], want={services_obj: v}, have={services_obj: h_service_item})
                    else:
                        temp_want = {'name': name, services_obj: v}
                        self.addcmd(temp_want, 'og_name', True)
                        self.compare([services_obj], want=temp_want, have={})
                if h_item and self.state in ['overridden', 'replaced']:
                    for k, v in iteritems(h_item):
                        temp_have = {'name': name, services_obj: v}
                        self.compare([services_obj], want={}, have=temp_have)
                if command_len < len(self.commands):
                    cmd = 'object-group service {0}'.format(name)
                    if cmd not in self.commands:
                        self.commands.insert(command_len, cmd)
            elif entry.get(port_obj):
                protocol = entry.get('protocol')
                if h_item:
                    h_item = self.convert_list_to_dict(val=h_item, source='source_port', destination='destination_port')
                entry = self.convert_list_to_dict(val=entry, source='source_port', destination='destination_port')
                command_len = len(self.commands)
                for k, v in iteritems(entry):
                    h_port_item = h_item.pop(k, {})
                    if 'http' in k and '_' in k:
                        temp = k.split('_')[0]
                        h_port_item = {temp: 'http'}
                    if h_port_item != v:
                        self.compare([port_obj], want={port_obj: v}, have={port_obj: h_port_item})
                    elif not h_port_item:
                        temp_want = {'name': name, port_obj: v}
                        self.compare([port_obj], want=temp_want, have={})
                if h_item and self.state in ['overridden', 'replaced']:
                    for k, v in iteritems(h_item):
                        temp_have = {'name': name, port_obj: v}
                        self.compare([port_obj], want={}, have=temp_have)
    self.check_for_have_and_overidden(have)