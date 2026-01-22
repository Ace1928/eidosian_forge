import calendar
import logging
from saml2 import SAMLError
from saml2 import class_name
from saml2 import extension_elements_to_elements
from saml2 import saml
from saml2 import samlp
from saml2 import time_util
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
from saml2.attribute_converter import to_local
from saml2.s_utils import RequestVersionTooHigh
from saml2.s_utils import RequestVersionTooLow
from saml2.saml import SCM_BEARER
from saml2.saml import SCM_HOLDER_OF_KEY
from saml2.saml import SCM_SENDER_VOUCHES
from saml2.saml import XSI_TYPE
from saml2.saml import attribute_from_string
from saml2.saml import encrypted_attribute_from_string
from saml2.samlp import STATUS_AUTHN_FAILED
from saml2.samlp import STATUS_INVALID_ATTR_NAME_OR_VALUE
from saml2.samlp import STATUS_INVALID_NAMEID_POLICY
from saml2.samlp import STATUS_NO_AUTHN_CONTEXT
from saml2.samlp import STATUS_NO_AVAILABLE_IDP
from saml2.samlp import STATUS_NO_PASSIVE
from saml2.samlp import STATUS_NO_SUPPORTED_IDP
from saml2.samlp import STATUS_PARTIAL_LOGOUT
from saml2.samlp import STATUS_PROXY_COUNT_EXCEEDED
from saml2.samlp import STATUS_REQUEST_DENIED
from saml2.samlp import STATUS_REQUEST_UNSUPPORTED
from saml2.samlp import STATUS_REQUEST_VERSION_DEPRECATED
from saml2.samlp import STATUS_REQUEST_VERSION_TOO_HIGH
from saml2.samlp import STATUS_REQUEST_VERSION_TOO_LOW
from saml2.samlp import STATUS_RESOURCE_NOT_RECOGNIZED
from saml2.samlp import STATUS_RESPONDER
from saml2.samlp import STATUS_TOO_MANY_RESPONSES
from saml2.samlp import STATUS_UNKNOWN_ATTR_PROFILE
from saml2.samlp import STATUS_UNKNOWN_PRINCIPAL
from saml2.samlp import STATUS_UNSUPPORTED_BINDING
from saml2.samlp import STATUS_VERSION_MISMATCH
from saml2.sigver import DecryptError
from saml2.sigver import SignatureError
from saml2.sigver import security_context
from saml2.sigver import signed
from saml2.time_util import later_than
from saml2.time_util import str_to_time
from saml2.validate import NotValid
from saml2.validate import valid_address
from saml2.validate import valid_instance
from saml2.validate import validate_before
from saml2.validate import validate_on_or_after
class StatusAuthnFailed(StatusError):
    pass