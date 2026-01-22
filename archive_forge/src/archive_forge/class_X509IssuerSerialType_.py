import saml2
from saml2 import SamlBase
class X509IssuerSerialType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:X509IssuerSerialType element"""
    c_tag = 'X509IssuerSerialType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}X509IssuerName'] = ('x509_issuer_name', X509IssuerName)
    c_children['{http://www.w3.org/2000/09/xmldsig#}X509SerialNumber'] = ('x509_serial_number', X509SerialNumber)
    c_child_order.extend(['x509_issuer_name', 'x509_serial_number'])

    def __init__(self, x509_issuer_name=None, x509_serial_number=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.x509_issuer_name = x509_issuer_name
        self.x509_serial_number = x509_serial_number