import saml2
from saml2 import SamlBase
def time_sync_token_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(TimeSyncTokenType_, xml_string)