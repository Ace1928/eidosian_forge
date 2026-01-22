import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class IDPListType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:protocol:IDPListType element"""
    c_tag = 'IDPListType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}IDPEntry'] = ('idp_entry', [IDPEntry])
    c_cardinality['idp_entry'] = {'min': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}GetComplete'] = ('get_complete', GetComplete)
    c_cardinality['get_complete'] = {'min': 0, 'max': 1}
    c_child_order.extend(['idp_entry', 'get_complete'])

    def __init__(self, idp_entry=None, get_complete=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.idp_entry = idp_entry or []
        self.get_complete = get_complete