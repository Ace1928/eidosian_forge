import saml2
from saml2 import SamlBase
class KeyInfoType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:KeyInfoType element"""
    c_tag = 'KeyInfoType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}KeyName'] = ('key_name', [KeyName])
    c_cardinality['key_name'] = {'min': 0}
    c_children['{http://www.w3.org/2000/09/xmldsig#}KeyValue'] = ('key_value', [KeyValue])
    c_cardinality['key_value'] = {'min': 0}
    c_children['{http://www.w3.org/2000/09/xmldsig#}RetrievalMethod'] = ('retrieval_method', [RetrievalMethod])
    c_cardinality['retrieval_method'] = {'min': 0}
    c_children['{http://www.w3.org/2000/09/xmldsig#}X509Data'] = ('x509_data', [X509Data])
    c_cardinality['x509_data'] = {'min': 0}
    c_children['{http://www.w3.org/2000/09/xmldsig#}PGPData'] = ('pgp_data', [PGPData])
    c_cardinality['pgp_data'] = {'min': 0}
    c_children['{http://www.w3.org/2000/09/xmldsig#}SPKIData'] = ('spki_data', [SPKIData])
    c_cardinality['spki_data'] = {'min': 0}
    c_children['{http://www.w3.org/2000/09/xmldsig#}MgmtData'] = ('mgmt_data', [MgmtData])
    c_cardinality['mgmt_data'] = {'min': 0}
    c_children['{http://www.w3.org/2000/09/xmlenc#}EncryptedKey'] = ('encrypted_key', None)
    c_cardinality['key_info'] = {'min': 0, 'max': 1}
    c_attributes['Id'] = ('id', 'ID', False)
    c_child_order.extend(['key_name', 'key_value', 'retrieval_method', 'x509_data', 'pgp_data', 'spki_data', 'mgmt_data', 'encrypted_key'])

    def __init__(self, key_name=None, key_value=None, retrieval_method=None, x509_data=None, pgp_data=None, spki_data=None, mgmt_data=None, encrypted_key=None, id=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.key_name = key_name or []
        self.key_value = key_value or []
        self.retrieval_method = retrieval_method or []
        self.x509_data = x509_data or []
        self.pgp_data = pgp_data or []
        self.spki_data = spki_data or []
        self.mgmt_data = mgmt_data or []
        self.encrypted_key = encrypted_key
        self.id = id