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
def _wrong_type_value(xsd, value):
    msg = 'Type and value do not match: {xsd}:{type}:{value}'
    msg = msg.format(xsd=xsd, type=type(value), value=value)
    raise ValueError(msg)