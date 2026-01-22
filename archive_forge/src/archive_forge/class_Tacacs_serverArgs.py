from __future__ import absolute_import, division, print_function
class Tacacs_serverArgs(object):
    """The arg spec for the sonic_tacacs_server module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'options': {'auth_type': {'choices': ['pap', 'chap', 'mschap', 'login'], 'default': 'pap', 'type': 'str'}, 'key': {'type': 'str', 'no_log': True}, 'servers': {'options': {'host': {'elements': 'dict', 'options': {'auth_type': {'choices': ['pap', 'chap', 'mschap', 'login'], 'default': 'pap', 'type': 'str'}, 'key': {'type': 'str', 'no_log': True}, 'name': {'type': 'str'}, 'port': {'default': 49, 'type': 'int'}, 'priority': {'default': 1, 'type': 'int'}, 'timeout': {'default': 5, 'type': 'int'}, 'vrf': {'default': 'default', 'type': 'str'}}, 'type': 'list'}}, 'type': 'dict'}, 'source_interface': {'type': 'str'}, 'timeout': {'type': 'int', 'default': 5}}, 'type': 'dict'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted'], 'default': 'merged'}}