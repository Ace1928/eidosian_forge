import ssl
import sys
import typing
def _set_ssl_context_verify_mode(ssl_context: ssl.SSLContext, verify_mode: ssl.VerifyMode) -> None:
    _original_super_SSLContext.verify_mode.__set__(ssl_context, verify_mode)