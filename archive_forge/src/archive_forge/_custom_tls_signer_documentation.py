import ctypes
import json
import logging
import os
import sys
import cffi  # type: ignore
import six
from google.auth import exceptions

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
        