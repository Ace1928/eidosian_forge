from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
class CorsError(enum.Enum):
    """
    The reason why request was blocked.
    """
    DISALLOWED_BY_MODE = 'DisallowedByMode'
    INVALID_RESPONSE = 'InvalidResponse'
    WILDCARD_ORIGIN_NOT_ALLOWED = 'WildcardOriginNotAllowed'
    MISSING_ALLOW_ORIGIN_HEADER = 'MissingAllowOriginHeader'
    MULTIPLE_ALLOW_ORIGIN_VALUES = 'MultipleAllowOriginValues'
    INVALID_ALLOW_ORIGIN_VALUE = 'InvalidAllowOriginValue'
    ALLOW_ORIGIN_MISMATCH = 'AllowOriginMismatch'
    INVALID_ALLOW_CREDENTIALS = 'InvalidAllowCredentials'
    CORS_DISABLED_SCHEME = 'CorsDisabledScheme'
    PREFLIGHT_INVALID_STATUS = 'PreflightInvalidStatus'
    PREFLIGHT_DISALLOWED_REDIRECT = 'PreflightDisallowedRedirect'
    PREFLIGHT_WILDCARD_ORIGIN_NOT_ALLOWED = 'PreflightWildcardOriginNotAllowed'
    PREFLIGHT_MISSING_ALLOW_ORIGIN_HEADER = 'PreflightMissingAllowOriginHeader'
    PREFLIGHT_MULTIPLE_ALLOW_ORIGIN_VALUES = 'PreflightMultipleAllowOriginValues'
    PREFLIGHT_INVALID_ALLOW_ORIGIN_VALUE = 'PreflightInvalidAllowOriginValue'
    PREFLIGHT_ALLOW_ORIGIN_MISMATCH = 'PreflightAllowOriginMismatch'
    PREFLIGHT_INVALID_ALLOW_CREDENTIALS = 'PreflightInvalidAllowCredentials'
    PREFLIGHT_MISSING_ALLOW_EXTERNAL = 'PreflightMissingAllowExternal'
    PREFLIGHT_INVALID_ALLOW_EXTERNAL = 'PreflightInvalidAllowExternal'
    PREFLIGHT_MISSING_ALLOW_PRIVATE_NETWORK = 'PreflightMissingAllowPrivateNetwork'
    PREFLIGHT_INVALID_ALLOW_PRIVATE_NETWORK = 'PreflightInvalidAllowPrivateNetwork'
    INVALID_ALLOW_METHODS_PREFLIGHT_RESPONSE = 'InvalidAllowMethodsPreflightResponse'
    INVALID_ALLOW_HEADERS_PREFLIGHT_RESPONSE = 'InvalidAllowHeadersPreflightResponse'
    METHOD_DISALLOWED_BY_PREFLIGHT_RESPONSE = 'MethodDisallowedByPreflightResponse'
    HEADER_DISALLOWED_BY_PREFLIGHT_RESPONSE = 'HeaderDisallowedByPreflightResponse'
    REDIRECT_CONTAINS_CREDENTIALS = 'RedirectContainsCredentials'
    INSECURE_PRIVATE_NETWORK = 'InsecurePrivateNetwork'
    INVALID_PRIVATE_NETWORK_ACCESS = 'InvalidPrivateNetworkAccess'
    UNEXPECTED_PRIVATE_NETWORK_ACCESS = 'UnexpectedPrivateNetworkAccess'
    NO_CORS_REDIRECT_MODE_NOT_FOLLOW = 'NoCorsRedirectModeNotFollow'
    PREFLIGHT_MISSING_PRIVATE_NETWORK_ACCESS_ID = 'PreflightMissingPrivateNetworkAccessId'
    PREFLIGHT_MISSING_PRIVATE_NETWORK_ACCESS_NAME = 'PreflightMissingPrivateNetworkAccessName'
    PRIVATE_NETWORK_ACCESS_PERMISSION_UNAVAILABLE = 'PrivateNetworkAccessPermissionUnavailable'
    PRIVATE_NETWORK_ACCESS_PERMISSION_DENIED = 'PrivateNetworkAccessPermissionDenied'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)