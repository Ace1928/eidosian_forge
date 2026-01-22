from __future__ import absolute_import, division, print_function
class Hsrp_interfacesArgs(object):
    """The arg spec for the nxos_hsrp_interfaces module"""
    argument_spec = {'running_config': {'type': 'str'}, 'config': {'type': 'list', 'elements': 'dict', 'options': {'name': {'type': 'str'}, 'bfd': {'choices': ['enable', 'disable'], 'type': 'str'}}}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'rendered', 'gathered', 'parsed'], 'default': 'merged', 'type': 'str'}}