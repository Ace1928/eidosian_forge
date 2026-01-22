import ctypes
import json
import logging
import os
import sys
import cffi  # type: ignore
import six
from google.auth import exceptions
class CustomTlsSigner(object):

    def __init__(self, enterprise_cert_file_path):
        """
        This class loads the offload and signer library, and calls APIs from
        these libraries to obtain the cert and a signing callback, and attach
        them to SSL context. The cert and the signing callback will be used
        for client authentication in TLS handshake.

        Args:
            enterprise_cert_file_path (str): the path to a enterprise cert JSON
                file. The file should contain the following field:

                    {
                        "libs": {
                            "ecp_client": "...",
                            "tls_offload": "..."
                        }
                    }
        """
        self._enterprise_cert_file_path = enterprise_cert_file_path
        self._cert = None
        self._sign_callback = None

    def load_libraries(self):
        try:
            with open(self._enterprise_cert_file_path, 'r') as f:
                enterprise_cert_json = json.load(f)
                libs = enterprise_cert_json['libs']
                signer_library = libs['ecp_client']
                offload_library = libs['tls_offload']
        except (KeyError, ValueError) as caught_exc:
            new_exc = exceptions.MutualTLSChannelError('enterprise cert file is invalid', caught_exc)
            six.raise_from(new_exc, caught_exc)
        self._offload_lib = load_offload_lib(offload_library)
        self._signer_lib = load_signer_lib(signer_library)

    def set_up_custom_key(self):
        self._cert = get_cert(self._signer_lib, self._enterprise_cert_file_path)
        self._sign_callback = get_sign_callback(self._signer_lib, self._enterprise_cert_file_path)

    def attach_to_ssl_context(self, ctx):
        if not self._offload_lib.ConfigureSslContext(self._sign_callback, ctypes.c_char_p(self._cert), _cast_ssl_ctx_to_void_p(ctx._ctx._context)):
            raise exceptions.MutualTLSChannelError('failed to configure SSL context')