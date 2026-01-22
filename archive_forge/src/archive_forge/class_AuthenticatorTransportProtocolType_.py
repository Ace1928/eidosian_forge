import saml2
from saml2 import SamlBase
class AuthenticatorTransportProtocolType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:AuthenticatorTransportProtocolType element"""
    c_tag = 'AuthenticatorTransportProtocolType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}HTTP'] = ('http', HTTP)
    c_cardinality['http'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}SSL'] = ('ssl', SSL)
    c_cardinality['ssl'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}MobileNetworkNoEncryption'] = ('mobile_network_no_encryption', MobileNetworkNoEncryption)
    c_cardinality['mobile_network_no_encryption'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}MobileNetworkRadioEncryption'] = ('mobile_network_radio_encryption', MobileNetworkRadioEncryption)
    c_cardinality['mobile_network_radio_encryption'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}MobileNetworkEndToEndEncryption'] = ('mobile_network_end_to_end_encryption', MobileNetworkEndToEndEncryption)
    c_cardinality['mobile_network_end_to_end_encryption'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}WTLS'] = ('wtls', WTLS)
    c_cardinality['wtls'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}IPSec'] = ('ip_sec', IPSec)
    c_cardinality['ip_sec'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}PSTN'] = ('pstn', PSTN)
    c_cardinality['pstn'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}ISDN'] = ('isdn', ISDN)
    c_cardinality['isdn'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}ADSL'] = ('adsl', ADSL)
    c_cardinality['adsl'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_child_order.extend(['http', 'ssl', 'mobile_network_no_encryption', 'mobile_network_radio_encryption', 'mobile_network_end_to_end_encryption', 'wtls', 'ip_sec', 'pstn', 'isdn', 'adsl', 'extension'])

    def __init__(self, http=None, ssl=None, mobile_network_no_encryption=None, mobile_network_radio_encryption=None, mobile_network_end_to_end_encryption=None, wtls=None, ip_sec=None, pstn=None, isdn=None, adsl=None, extension=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.http = http
        self.ssl = ssl
        self.mobile_network_no_encryption = mobile_network_no_encryption
        self.mobile_network_radio_encryption = mobile_network_radio_encryption
        self.mobile_network_end_to_end_encryption = mobile_network_end_to_end_encryption
        self.wtls = wtls
        self.ip_sec = ip_sec
        self.pstn = pstn
        self.isdn = isdn
        self.adsl = adsl
        self.extension = extension or []