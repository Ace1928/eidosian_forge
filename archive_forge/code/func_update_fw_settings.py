from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.posix.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import missing_required_lib
import_failure = True
def update_fw_settings(self, fw_zone, fw_settings):
    if self.fw_offline:
        self.fw.config.set_zone_config(fw_zone, fw_settings.settings)
    else:
        fw_zone.update(fw_settings)