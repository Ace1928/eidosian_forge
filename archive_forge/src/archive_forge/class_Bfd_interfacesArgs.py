from __future__ import absolute_import, division, print_function
class Bfd_interfacesArgs(object):
    """The arg spec for the nxos_bfd_interfaces module"""
    argument_spec = {'running_config': {'type': 'str'}, 'config': {'elements': 'dict', 'options': {'name': {'type': 'str'}, 'bfd': {'choices': ['enable', 'disable'], 'type': 'str'}, 'echo': {'choices': ['enable', 'disable'], 'type': 'str'}}, 'type': 'list'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'gathered', 'rendered', 'parsed'], 'default': 'merged', 'type': 'str'}}