from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def update_autoload(self):
    """Update automatic load balancing state."""
    current_configuration = self.get_current_configuration()
    if self.autoload_enabled and (not current_configuration['host_connectivity_reporting_enabled']):
        try:
            rc, host_connectivity_reporting = self.request('storage-systems/%s/symbol/setHostConnectivityReporting?verboseErrorResponse=true' % self.ssid, method='POST', data={'enableHostConnectivityReporting': self.autoload_enabled})
        except Exception as error:
            self.module.fail_json(msg='Failed to enable host connectivity reporting which is needed for automatic load balancing state. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
    try:
        rc, autoload = self.request('storage-systems/%s/symbol/setAutoLoadBalancing?verboseErrorResponse=true' % self.ssid, method='POST', data={'enableAutoLoadBalancing': self.autoload_enabled})
    except Exception as error:
        self.module.fail_json(msg='Failed to set automatic load balancing state. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))