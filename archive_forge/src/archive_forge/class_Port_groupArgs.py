from __future__ import absolute_import, division, print_function
class Port_groupArgs(object):
    """The arg spec for the sonic_port_group module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'id': {'required': True, 'type': 'str'}, 'speed': {'choices': ['SPEED_10MB', 'SPEED_100MB', 'SPEED_1GB', 'SPEED_2500MB', 'SPEED_5GB', 'SPEED_10GB', 'SPEED_20GB', 'SPEED_25GB', 'SPEED_40GB', 'SPEED_50GB', 'SPEED_100GB', 'SPEED_400GB'], 'type': 'str'}}, 'type': 'list'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted'], 'default': 'merged', 'type': 'str'}}