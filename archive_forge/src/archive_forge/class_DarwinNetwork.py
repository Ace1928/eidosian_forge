from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.facts.network.base import NetworkCollector
from ansible.module_utils.facts.network.generic_bsd import GenericBsdIfconfigNetwork
class DarwinNetwork(GenericBsdIfconfigNetwork):
    """
    This is the Mac macOS Darwin Network Class.
    It uses the GenericBsdIfconfigNetwork unchanged
    """
    platform = 'Darwin'

    def parse_media_line(self, words, current_if, ips):
        current_if['media'] = 'Unknown'
        current_if['media_select'] = words[1]
        if len(words) > 2:
            if words[1] == '<unknown' and words[2] == 'type>':
                current_if['media_select'] = 'Unknown'
                current_if['media_type'] = 'unknown type'
            else:
                current_if['media_type'] = words[2][1:-1]
        if len(words) > 3:
            current_if['media_options'] = self.get_options(words[3])