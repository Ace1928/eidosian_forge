from os_ken.lib.of_config.base import _Base, _e, _ct
class OFPortStateType(_Base):
    _ELEMENTS = [_e('oper-state', is_list=False), _e('blocked', is_list=False), _e('live', is_list=False)]