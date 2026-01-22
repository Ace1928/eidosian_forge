import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class ScopingType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:protocol:ScopingType element"""
    c_tag = 'ScopingType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}IDPList'] = ('idp_list', IDPList)
    c_cardinality['idp_list'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}RequesterID'] = ('requester_id', [RequesterID])
    c_cardinality['requester_id'] = {'min': 0}
    c_attributes['ProxyCount'] = ('proxy_count', 'nonNegativeInteger', False)
    c_child_order.extend(['idp_list', 'requester_id'])

    def __init__(self, idp_list=None, requester_id=None, proxy_count=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.idp_list = idp_list
        self.requester_id = requester_id or []
        self.proxy_count = proxy_count