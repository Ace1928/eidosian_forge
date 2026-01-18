import saml2
from saml2 import SamlBase
def x509_data_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(X509DataType_, xml_string)