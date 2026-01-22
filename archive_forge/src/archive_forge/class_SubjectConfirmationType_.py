import base64
from datetime import date
from datetime import datetime
import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
from saml2.validate import MustValueError
from saml2.validate import ShouldValueError
from saml2.validate import valid_domain_name
from saml2.validate import valid_ipv4
from saml2.validate import valid_ipv6
class SubjectConfirmationType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:assertion:SubjectConfirmationType
    element"""
    c_tag = 'SubjectConfirmationType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}BaseID'] = ('base_id', BaseID)
    c_cardinality['base_id'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}NameID'] = ('name_id', NameID)
    c_cardinality['name_id'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}EncryptedID'] = ('encrypted_id', EncryptedID)
    c_cardinality['encrypted_id'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}SubjectConfirmationData'] = ('subject_confirmation_data', SubjectConfirmationData)
    c_cardinality['subject_confirmation_data'] = {'min': 0, 'max': 1}
    c_attributes['Method'] = ('method', 'anyURI', True)
    c_child_order.extend(['base_id', 'name_id', 'encrypted_id', 'subject_confirmation_data'])

    def __init__(self, base_id=None, name_id=None, encrypted_id=None, subject_confirmation_data=None, method=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.base_id = base_id
        self.name_id = name_id
        self.encrypted_id = encrypted_id
        self.subject_confirmation_data = subject_confirmation_data
        self.method = method