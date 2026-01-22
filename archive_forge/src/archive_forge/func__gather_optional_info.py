from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
def _gather_optional_info(self, options, enclosure):
    enclosure_client = self.oneview_client.enclosures
    info = {}
    if options.get('script'):
        info['enclosure_script'] = enclosure_client.get_script(enclosure['uri'])
    if options.get('environmentalConfiguration'):
        env_config = enclosure_client.get_environmental_configuration(enclosure['uri'])
        info['enclosure_environmental_configuration'] = env_config
    if options.get('utilization'):
        info['enclosure_utilization'] = self._get_utilization(enclosure, options['utilization'])
    return info