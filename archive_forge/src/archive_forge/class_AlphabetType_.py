import saml2
from saml2 import SamlBase
class AlphabetType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:AlphabetType element"""
    c_tag = 'AlphabetType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['requiredChars'] = ('required_chars', 'string', True)
    c_attributes['excludedChars'] = ('excluded_chars', 'string', False)
    c_attributes['case'] = ('case', 'string', False)

    def __init__(self, required_chars=None, excluded_chars=None, case=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.required_chars = required_chars
        self.excluded_chars = excluded_chars
        self.case = case