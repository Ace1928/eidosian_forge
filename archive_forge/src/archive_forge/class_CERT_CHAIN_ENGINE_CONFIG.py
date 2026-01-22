import contextlib
import ssl
import typing
from ctypes import WinDLL  # type: ignore
from ctypes import WinError  # type: ignore
from ctypes import (
from ctypes.wintypes import (
from typing import TYPE_CHECKING, Any
from ._ssl_constants import _set_ssl_context_verify_mode
class CERT_CHAIN_ENGINE_CONFIG(Structure):
    _fields_ = (('cbSize', DWORD), ('hRestrictedRoot', HCERTSTORE), ('hRestrictedTrust', HCERTSTORE), ('hRestrictedOther', HCERTSTORE), ('cAdditionalStore', DWORD), ('rghAdditionalStore', c_void_p), ('dwFlags', DWORD), ('dwUrlRetrievalTimeout', DWORD), ('MaximumCachedCertificates', DWORD), ('CycleDetectionModulus', DWORD), ('hExclusiveRoot', HCERTSTORE), ('hExclusiveTrustedPeople', HCERTSTORE), ('dwExclusiveFlags', DWORD))