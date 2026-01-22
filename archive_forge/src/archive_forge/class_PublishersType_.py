import saml2
from saml2 import SamlBase
from saml2 import md
class PublishersType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:dri:PublishersType element"""
    c_tag = 'PublishersType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata:dri}Publisher'] = ('publisher', [Publisher])
    c_cardinality['publisher'] = {'min': 0}
    c_child_order.extend(['publisher'])

    def __init__(self, publisher=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.publisher = publisher or []