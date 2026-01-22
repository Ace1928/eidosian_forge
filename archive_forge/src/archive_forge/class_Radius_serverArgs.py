from __future__ import absolute_import, division, print_function
class Radius_serverArgs(object):
    """The arg spec for the sonic_radius_server module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'options': {'auth_type': {'choices': ['pap', 'chap', 'mschapv2'], 'default': 'pap', 'type': 'str'}, 'key': {'type': 'str', 'no_log': True}, 'nas_ip': {'type': 'str'}, 'retransmit': {'type': 'int'}, 'servers': {'options': {'host': {'elements': 'dict', 'options': {'auth_type': {'choices': ['pap', 'chap', 'mschapv2'], 'type': 'str'}, 'key': {'type': 'str', 'no_log': True}, 'name': {'type': 'str'}, 'port': {'type': 'int', 'default': 1812}, 'priority': {'type': 'int'}, 'retransmit': {'type': 'int'}, 'source_interface': {'type': 'str'}, 'timeout': {'type': 'int'}, 'vrf': {'type': 'str'}}, 'type': 'list'}}, 'type': 'dict'}, 'statistics': {'type': 'bool'}, 'timeout': {'type': 'int', 'default': 5}}, 'type': 'dict'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted'], 'default': 'merged'}}