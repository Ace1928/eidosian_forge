import saml2
from saml2 import SamlBase
def token_from_string(xml_string):
    return saml2.create_class_from_xml_string(Token, xml_string)