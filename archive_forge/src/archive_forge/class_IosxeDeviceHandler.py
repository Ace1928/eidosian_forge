from .default import DefaultDeviceHandler
from ncclient.operations.third_party.iosxe.rpc import SaveConfig
from ncclient.xml_ import BASE_NS_1_0
import logging
class IosxeDeviceHandler(DefaultDeviceHandler):
    """
    Cisco IOS-XE handler for device specific information.

    """

    def __init__(self, device_params):
        super(IosxeDeviceHandler, self).__init__(device_params)

    def add_additional_operations(self):
        dict = {}
        dict['save_config'] = SaveConfig
        return dict

    def add_additional_ssh_connect_params(self, kwargs):
        kwargs['unknown_host_cb'] = iosxe_unknown_host_cb

    def transform_edit_config(self, node):
        nodes = node.findall('./config')
        if len(nodes) == 1:
            logger.debug('IOS XE handler: patching namespace of config element')
            nodes[0].tag = '{%s}%s' % (BASE_NS_1_0, 'config')
        return node