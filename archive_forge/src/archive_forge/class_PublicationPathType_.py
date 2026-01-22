import saml2
from saml2 import SamlBase
from saml2 import md
class PublicationPathType_(SamlBase):
    """The urn:oasis:names:tc:SAML:metadata:rpi:PublicationPathType element"""
    c_tag = 'PublicationPathType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:metadata:rpi}Publication'] = ('publication', [Publication])
    c_cardinality['publication'] = {'min': 0}
    c_child_order.extend(['publication'])

    def __init__(self, publication=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.publication = publication or []