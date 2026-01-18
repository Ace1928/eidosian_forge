import saml2
from saml2 import SamlBase
def x509_issuer_serial_from_string(xml_string):
    return saml2.create_class_from_xml_string(X509IssuerSerial, xml_string)