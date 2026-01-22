from __future__ import (absolute_import, division, print_function)
class FactsArgs(object):
    """ The arg spec for the fortios monitor module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'host': {'required': False, 'type': 'str'}, 'username': {'required': False, 'type': 'str'}, 'password': {'required': False, 'type': 'str', 'no_log': True}, 'vdom': {'required': False, 'type': 'str', 'default': 'root'}, 'https': {'required': False, 'type': 'bool', 'default': True}, 'ssl_verify': {'required': False, 'type': 'bool', 'default': False}, 'gather_subset': {'required': True, 'type': 'list', 'elements': 'dict', 'options': {'fact': {'required': True, 'type': 'str'}, 'filters': {'required': False, 'type': 'list', 'elements': 'dict'}}}}