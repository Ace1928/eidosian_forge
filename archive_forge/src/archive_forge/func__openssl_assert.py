from __future__ import annotations
import os
import sys
import threading
import types
import typing
import warnings
import cryptography
from cryptography.exceptions import InternalError
from cryptography.hazmat.bindings._rust import _openssl, openssl
from cryptography.hazmat.bindings.openssl._conditional import CONDITIONAL_NAMES
def _openssl_assert(lib, ok: bool, errors: typing.Optional[typing.List[openssl.OpenSSLError]]=None) -> None:
    if not ok:
        if errors is None:
            errors = openssl.capture_error_stack()
        raise InternalError('Unknown OpenSSL error. This error is commonly encountered when another library is not cleaning up the OpenSSL error stack. If you are using cryptography with another library that uses OpenSSL try disabling it before reporting a bug. Otherwise please file an issue at https://github.com/pyca/cryptography/issues with information on how to reproduce this. ({!r})'.format(errors), errors)