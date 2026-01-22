from __future__ import absolute_import, division, print_function
class PkiArgs(object):
    """The arg spec for the sonic_pki module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'options': {'security_profiles': {'elements': 'dict', 'options': {'cdp_list': {'elements': 'str', 'type': 'list'}, 'certificate_name': {'type': 'str'}, 'key_usage_check': {'type': 'bool'}, 'ocsp_responder_list': {'elements': 'str', 'type': 'list'}, 'peer_name_check': {'type': 'bool'}, 'profile_name': {'required': True, 'type': 'str'}, 'revocation_check': {'type': 'bool'}, 'trust_store': {'type': 'str'}}, 'type': 'list'}, 'trust_stores': {'elements': 'dict', 'options': {'ca_name': {'elements': 'str', 'type': 'list'}, 'name': {'required': True, 'type': 'str'}}, 'type': 'list'}}, 'type': 'dict'}, 'state': {'choices': ['merged', 'deleted', 'replaced', 'overridden'], 'default': 'merged', 'type': 'str'}}