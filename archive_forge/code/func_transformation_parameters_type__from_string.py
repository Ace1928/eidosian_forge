import saml2
from saml2 import SamlBase
def transformation_parameters_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(TransformationParametersType_, xml_string)