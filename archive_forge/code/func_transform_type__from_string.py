import saml2
from saml2 import SamlBase
def transform_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(TransformType_, xml_string)