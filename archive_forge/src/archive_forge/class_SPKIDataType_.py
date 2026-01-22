import saml2
from saml2 import SamlBase
class SPKIDataType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:SPKIDataType element"""
    c_tag = 'SPKIDataType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}SPKISexp'] = ('spki_sexp', [SPKISexp])
    c_cardinality['spki_sexp'] = {'min': 1}
    c_child_order.extend(['spki_sexp'])

    def __init__(self, spki_sexp=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.spki_sexp = spki_sexp or []