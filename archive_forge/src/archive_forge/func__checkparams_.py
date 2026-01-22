from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def _checkparams_(self):
    """check all input params"""
    if self.function_flag == 'singleBFD':
        if not self.next_hop:
            self.module.fail_json(msg='Error: missing required argument: next_hop.')
        if self.state != 'absent':
            if self.nhp_interface == 'Invalid0' and (not self.prefix or self.prefix == '0.0.0.0'):
                self.module.fail_json(msg='Error: If a nhp_interface is not configured, the prefix must be configured.')
    if self.function_flag != 'globalBFD':
        if self.function_flag == 'dynamicBFD' or self.function_flag == 'staticBFD':
            if not self.mask:
                self.module.fail_json(msg='Error: missing required argument: mask.')
            if not self.mask.isdigit():
                self.module.fail_json(msg='Error: Mask is invalid.')
        if self.function_flag != 'singleBFD' or (self.function_flag == 'singleBFD' and self.destvrf != '_public_'):
            if not self.prefix:
                self.module.fail_json(msg='Error: missing required argument: prefix.')
            if not self._convertipprefix_():
                self.module.fail_json(msg='Error: The %s is not a valid address' % self.prefix)
        if self.nhp_interface != 'Invalid0' and self.destvrf != '_public_':
            self.module.fail_json(msg='Error: Destination vrf dose not support next hop is interface.')
        if not self.next_hop and self.nhp_interface == 'Invalid0':
            self.module.fail_json(msg='Error: one of the following is required: next_hop,nhp_interface.')
    if self.function_flag == 'dynamicBFD' or self.function_flag == 'staticBFD':
        if self.description:
            if not is_valid_description(self.description):
                self.module.fail_json(msg='Error: Dsecription length should be 1 - 35, and can not contain "?".')
        if self.tag is not None:
            if not is_valid_tag(self.tag):
                self.module.fail_json(msg='Error: Tag should be integer 1 - 4294967295.')
        if self.pref is not None:
            if not is_valid_preference(self.pref):
                self.module.fail_json(msg='Error: Preference should be integer 1 - 255.')
        if self.function_flag == 'staticBFD':
            if self.bfd_session_name:
                if not is_valid_bdf_session_name(self.bfd_session_name):
                    self.module.fail_json(msg='Error: bfd_session_name length should be 1 - 15, and can not contain Space.')
    if self.aftype == 'v4':
        if self.function_flag == 'dynamicBFD' or self.function_flag == 'staticBFD':
            if int(self.mask) > 32 or int(self.mask) < 0:
                self.module.fail_json(msg='Error: Ipv4 mask must be an integer between 1 and 32.')
        if self.function_flag != 'globalBFD':
            if self.next_hop:
                if not is_valid_v4addr(self.next_hop):
                    self.module.fail_json(msg='Error: The %s is not a valid address.' % self.next_hop)
    if self.aftype == 'v6':
        if self.function_flag == 'dynamicBFD' or self.function_flag == 'staticBFD':
            if int(self.mask) > 128 or int(self.mask) < 0:
                self.module.fail_json(msg='Error: Ipv6 mask must be an integer between 1 and 128.')
        if self.function_flag != 'globalBFD':
            if self.next_hop:
                if not is_valid_v6addr(self.next_hop):
                    self.module.fail_json(msg='Error: The %s is not a valid address.' % self.next_hop)
    if self.function_flag == 'globalBFD' or self.function_flag == 'singleBFD':
        if self.min_tx_interval:
            if not is_valid_bdf_interval(self.min_tx_interval):
                self.module.fail_json(msg='Error: min_tx_interval should be integer 50 - 1000.')
        if self.min_rx_interval:
            if not is_valid_bdf_interval(self.min_rx_interval):
                self.module.fail_json(msg='Error: min_rx_interval should be integer 50 - 1000.')
        if self.detect_multiplier:
            if not is_valid_bdf_multiplier(self.detect_multiplier):
                self.module.fail_json(msg='Error: detect_multiplier should be integer 3 - 50.')
        if self.function_flag == 'globalBFD':
            if self.state != 'absent':
                if not self.min_tx_interval and (not self.min_rx_interval) and (not self.detect_multiplier):
                    self.module.fail_json(msg='Error: one of the following is required: min_tx_interval,detect_multiplier,min_rx_interval.')
            else:
                if not self.commands:
                    self.module.fail_json(msg='Error: missing required argument: command.')
                if compare_command(self.commands):
                    self.module.fail_json(msg='Error: The command %s line is incorrect.' % ','.join(self.commands))