from os_ken.lib.of_config.base import _Base, _e, _ct
class OFCapableSwitchType(_Base):
    _ELEMENTS = [_e('id', is_list=False), _e('config-version', is_list=False), _ct('configuration-points', None, is_list=False), _ct('resources', OFCapableSwitchResourcesType, is_list=False), _ct('logical-switches', OFCapableSwitchLogicalSwitchesType, is_list=False)]