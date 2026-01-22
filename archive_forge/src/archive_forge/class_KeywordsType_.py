import saml2
from saml2 import SamlBase
from saml2 import md
class KeywordsType_(ListOfStrings_):
    """The urn:oasis:names:tc:SAML:metadata:ui:KeywordsType element"""
    c_tag = 'KeywordsType'
    c_namespace = NAMESPACE
    c_children = ListOfStrings_.c_children.copy()
    c_attributes = ListOfStrings_.c_attributes.copy()
    c_child_order = ListOfStrings_.c_child_order[:]
    c_cardinality = ListOfStrings_.c_cardinality.copy()
    c_attributes['{http://www.w3.org/XML/1998/namespace}lang'] = ('lang', 'mdui:listOfStrings', True)

    def __init__(self, lang=None, text=None, extension_elements=None, extension_attributes=None):
        ListOfStrings_.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.lang = lang