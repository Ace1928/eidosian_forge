from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
import time
def wait_for_ilo_reboot_completion(self, polling_interval=60, max_polling_time=1800):
    time.sleep(10)
    state = self.get_server_poststate()
    if not state['ret']:
        return state
    count = int(max_polling_time / polling_interval)
    times = 0
    pcount = 0
    while state['server_poststate'] in ['PowerOff', 'Off'] and pcount < 5:
        time.sleep(10)
        state = self.get_server_poststate()
        if not state['ret']:
            return state
        if state['server_poststate'] not in ['PowerOff', 'Off']:
            break
        pcount = pcount + 1
    if state['server_poststate'] in ['PowerOff', 'Off']:
        return {'ret': False, 'changed': False, 'msg': 'Server is powered OFF'}
    if state['server_poststate'] in ['InPostDiscoveryComplete', 'FinishedPost']:
        return {'ret': True, 'changed': False, 'msg': 'Server is not rebooting'}
    while state['server_poststate'] not in ['InPostDiscoveryComplete', 'FinishedPost'] and count > times:
        state = self.get_server_poststate()
        if not state['ret']:
            return state
        if state['server_poststate'] in ['InPostDiscoveryComplete', 'FinishedPost']:
            return {'ret': True, 'changed': True, 'msg': 'Server reboot is completed'}
        time.sleep(polling_interval)
        times = times + 1
    return {'ret': False, 'changed': False, 'msg': 'Server Reboot has failed, server state: {state} '.format(state=state)}