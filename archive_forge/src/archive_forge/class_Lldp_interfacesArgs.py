from __future__ import absolute_import, division, print_function
class Lldp_interfacesArgs(object):
    """The arg spec for the vyos_lldp_interfaces module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'enable': {'default': True, 'type': 'bool'}, 'location': {'options': {'civic_based': {'options': {'ca_info': {'elements': 'dict', 'options': {'ca_type': {'type': 'int'}, 'ca_value': {'type': 'str'}}, 'type': 'list'}, 'country_code': {'required': True, 'type': 'str'}}, 'type': 'dict'}, 'coordinate_based': {'options': {'altitude': {'type': 'int'}, 'datum': {'choices': ['WGS84', 'NAD83', 'MLLW'], 'type': 'str'}, 'latitude': {'required': True, 'type': 'str'}, 'longitude': {'required': True, 'type': 'str'}}, 'type': 'dict'}, 'elin': {'type': 'str'}}, 'type': 'dict'}, 'name': {'required': True, 'type': 'str'}}, 'type': 'list'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'rendered', 'gathered', 'parsed'], 'default': 'merged', 'type': 'str'}}