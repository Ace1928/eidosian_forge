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
class ConditionsType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:assertion:ConditionsType element"""
    c_tag = 'ConditionsType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Condition'] = ('condition', [Condition])
    c_cardinality['condition'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AudienceRestriction'] = ('audience_restriction', [AudienceRestriction])
    c_cardinality['audience_restriction'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}OneTimeUse'] = ('one_time_use', [OneTimeUse])
    c_cardinality['one_time_use'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}ProxyRestriction'] = ('proxy_restriction', [ProxyRestriction])
    c_cardinality['proxy_restriction'] = {'min': 0}
    c_attributes['NotBefore'] = ('not_before', 'dateTime', False)
    c_attributes['NotOnOrAfter'] = ('not_on_or_after', 'dateTime', False)
    c_child_order.extend(['condition', 'audience_restriction', 'one_time_use', 'proxy_restriction'])

    def __init__(self, condition=None, audience_restriction=None, one_time_use=None, proxy_restriction=None, not_before=None, not_on_or_after=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.condition = condition or []
        self.audience_restriction = audience_restriction or []
        self.one_time_use = one_time_use or []
        self.proxy_restriction = proxy_restriction or []
        self.not_before = not_before
        self.not_on_or_after = not_on_or_after

    def verify(self):
        if self.one_time_use:
            if len(self.one_time_use) != 1:
                raise Exception('Cannot be used more than once')
        if self.proxy_restriction:
            if len(self.proxy_restriction) != 1:
                raise Exception('Cannot be used more than once')
        return SamlBase.verify(self)