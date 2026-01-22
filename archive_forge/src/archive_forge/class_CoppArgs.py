from __future__ import absolute_import, division, print_function
class CoppArgs(object):
    """The arg spec for the sonic_copp module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'options': {'copp_groups': {'elements': 'dict', 'options': {'cbs': {'type': 'str'}, 'cir': {'type': 'str'}, 'copp_name': {'required': True, 'type': 'str'}, 'queue': {'type': 'int'}, 'trap_action': {'type': 'str'}, 'trap_priority': {'type': 'int'}}, 'type': 'list'}}, 'type': 'dict'}, 'state': {'choices': ['merged', 'deleted', 'replaced', 'overridden'], 'default': 'merged', 'type': 'str'}}