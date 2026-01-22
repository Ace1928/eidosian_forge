import saml2
from saml2 import SamlBase
class PGPDataType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:PGPDataType element"""
    c_tag = 'PGPDataType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}PGPKeyID'] = ('pgp_key_id', PGPKeyID)
    c_children['{http://www.w3.org/2000/09/xmldsig#}PGPKeyPacket'] = ('pgp_key_packet', PGPKeyPacket)
    c_cardinality['pgp_key_packet'] = {'min': 0, 'max': 1}
    c_child_order.extend(['pgp_key_id', 'pgp_key_packet'])

    def __init__(self, pgp_key_id=None, pgp_key_packet=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.pgp_key_id = pgp_key_id
        self.pgp_key_packet = pgp_key_packet