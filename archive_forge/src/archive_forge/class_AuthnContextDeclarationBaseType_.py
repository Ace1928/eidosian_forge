import saml2
from saml2 import SamlBase
class AuthnContextDeclarationBaseType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:AuthnContextDeclarationBaseType element"""
    c_tag = 'AuthnContextDeclarationBaseType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Identification'] = ('identification', Identification)
    c_cardinality['identification'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}TechnicalProtection'] = ('technical_protection', TechnicalProtection)
    c_cardinality['technical_protection'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}OperationalProtection'] = ('operational_protection', OperationalProtection)
    c_cardinality['operational_protection'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}AuthnMethod'] = ('authn_method', AuthnMethod)
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}GoverningAgreements'] = ('governing_agreements', GoverningAgreements)
    c_cardinality['governing_agreements'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_attributes['ID'] = ('id', 'ID', False)
    c_child_order.extend(['identification', 'technical_protection', 'operational_protection', 'authn_method', 'governing_agreements', 'extension'])

    def __init__(self, identification=None, technical_protection=None, operational_protection=None, authn_method=None, governing_agreements=None, extension=None, id=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.identification = identification
        self.technical_protection = technical_protection
        self.operational_protection = operational_protection
        self.authn_method = authn_method
        self.governing_agreements = governing_agreements
        self.extension = extension or []
        self.id = id