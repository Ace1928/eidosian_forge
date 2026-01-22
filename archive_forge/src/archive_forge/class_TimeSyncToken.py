import saml2
from saml2 import SamlBase
class TimeSyncToken(TimeSyncTokenType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:TimeSyncToken element"""
    c_tag = 'TimeSyncToken'
    c_namespace = NAMESPACE
    c_children = TimeSyncTokenType_.c_children.copy()
    c_attributes = TimeSyncTokenType_.c_attributes.copy()
    c_child_order = TimeSyncTokenType_.c_child_order[:]
    c_cardinality = TimeSyncTokenType_.c_cardinality.copy()