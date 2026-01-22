from os_ken.lib.of_config.base import _Base, _e, _ct
class OFQueueType(_Base):
    _ELEMENTS = [_e('resource-id', is_list=False), _e('id', is_list=False), _e('port', is_list=False), _ct('properties', OFQueuePropertiesType, is_list=False)]