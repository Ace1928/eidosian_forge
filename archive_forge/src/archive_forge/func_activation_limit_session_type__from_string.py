import saml2
from saml2 import SamlBase
def activation_limit_session_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(ActivationLimitSessionType_, xml_string)