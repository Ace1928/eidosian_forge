from os_ken.lib.of_config.base import _Base, _e, _ct
class OFCapableSwitchResourcesType(_Base):
    _ELEMENTS = [_ct('port', OFPortType, is_list=True), _ct('queue', OFQueueType, is_list=True), _ct('owned-certificate', None, is_list=True), _ct('external-certificate', None, is_list=True), _ct('flow-table', None, is_list=True)]