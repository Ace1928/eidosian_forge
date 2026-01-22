import saml2
from saml2 import SamlBase
class TimestampType_(SamlBase):
    """The http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd:TimestampType element"""
    c_tag = 'TimestampType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}Created'] = ('created', Created)
    c_cardinality['created'] = {'min': 0, 'max': 1}
    c_children['{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}Expires'] = ('expires', Expires)
    c_cardinality['expires'] = {'min': 0, 'max': 1}
    c_attributes['Id'] = ('Id', 'anyURI', False)
    c_child_order.extend(['created', 'expires'])

    def __init__(self, created=None, expires=None, Id=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.created = created
        self.expires = expires
        self.Id = Id