import saml2
from saml2 import SamlBase
class FaultCodesType_(SamlBase):
    """The http://www.w3.org/2005/08/addressing:FaultCodesType element"""
    c_tag = 'FaultCodesType'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'xs:QName', 'enumeration': ['tns:InvalidAddressingHeader', 'tns:InvalidAddress', 'tns:InvalidEPR', 'tns:InvalidCardinality', 'tns:MissingAddressInEPR', 'tns:DuplicateMessageID', 'tns:ActionMismatch', 'tns:MessageAddressingHeaderRequired', 'tns:DestinationUnreachable', 'tns:ActionNotSupported', 'tns:EndpointUnavailable']}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()