from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospfv3 import (
def _address_family_compare(self, want, have):
    if want['process_id'] == have.get('process_id') or want['process_id']:
        af_parsers = ['address_family.adjacency', 'address_family.auto_cost', 'address_family.bfd', 'address_family.capability', 'address_family.compatible', 'address_family.default_information', 'address_family.default_metric', 'address_family.distance.admin_distance', 'address_family.distance.ospf', 'address_family.distribute_list.acls', 'address_family.distribute_list.prefix', 'address_family.distribute_list.route_map', 'address_family.event_log', 'address_family.graceful_restart', 'address_family.interface_id', 'address_family.limit', 'address_family.local_rib_criteria', 'address_family.log_adjacency_changes', 'address_family.manet', 'address_family.max_lsa', 'address_family.max_metric', 'address_family.maximum_paths', 'address_family.passive_interface', 'address_family.prefix_suppression', 'address_family.queue_depth.hello', 'address_family.queue_depth.update', 'address_family.router_id', 'address_family.shutdown', 'address_family.summary_prefix', 'address_family.timers.throttle.lsa', 'address_family.timers.throttle.spf']
        delete_exit_family = False
        for each_want_af in want['address_family']:
            if have.get('address_family'):
                for each_have_af in have['address_family']:
                    if each_have_af.get('vrf') == each_want_af.get('vrf') and each_have_af.get('afi') == each_want_af.get('afi'):
                        self.compare(parsers=['address_family'], want={'address_family': each_want_af}, have={'address_family': each_have_af})
                        self.compare(parsers=af_parsers, want=each_want_af, have=each_have_af)
                    elif each_have_af.get('afi') == each_want_af.get('afi'):
                        self.compare(parsers=['address_family'], want={'address_family': each_want_af}, have={'address_family': each_have_af})
                        self.compare(parsers=af_parsers, want={'address_family': each_want_af}, have={'address_family': each_have_af})
                    if each_want_af.get('areas'):
                        af_want_areas = {}
                        af_have_areas = {}
                        for each_area in each_want_af['areas']:
                            af_want_areas.update({each_area['area_id']: each_area})
                        if each_have_af.get('areas'):
                            for each_area in each_have_af['areas']:
                                af_have_areas.update({each_area['area_id']: each_area})
                        if 'exit-address-family' in self.commands:
                            del self.commands[self.commands.index('exit-address-family')]
                            delete_exit_family = True
                        if af_have_areas:
                            self._areas_compare({'areas': af_want_areas}, {'areas': af_have_areas})
                        else:
                            self._areas_compare({'areas': af_want_areas}, dict())
                        if delete_exit_family:
                            self.commands.append('exit-address-family')
            else:
                temp_cmd_before = self.commands
                self.commands = []
                self.compare(parsers=['address_family'], want={'address_family': each_want_af}, have=dict())
                self.compare(parsers=af_parsers, want=each_want_af, have=dict())
                if each_want_af.get('areas'):
                    af_areas = {}
                    for each_area in each_want_af['areas']:
                        af_areas.update({each_area['area_id']: each_area})
                    self._areas_compare({'areas': af_areas}, dict())
                del self.commands[self.commands.index('exit-address-family')]
                self.commands.append('exit-address-family')
                self.commands[0:0] = temp_cmd_before