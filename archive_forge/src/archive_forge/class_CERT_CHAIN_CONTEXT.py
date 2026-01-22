import contextlib
import ssl
import typing
from ctypes import WinDLL  # type: ignore
from ctypes import WinError  # type: ignore
from ctypes import (
from ctypes.wintypes import (
from typing import TYPE_CHECKING, Any
from ._ssl_constants import _set_ssl_context_verify_mode
class CERT_CHAIN_CONTEXT(Structure):
    _fields_ = (('cbSize', DWORD), ('TrustStatus', CERT_TRUST_STATUS), ('cChain', DWORD), ('rgpChain', POINTER(PCERT_SIMPLE_CHAIN)), ('cLowerQualityChainContext', DWORD), ('rgpLowerQualityChainContext', c_void_p), ('fHasRevocationFreshnessTime', BOOL), ('dwRevocationFreshnessTime', DWORD))