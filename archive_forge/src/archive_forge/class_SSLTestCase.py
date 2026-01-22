import asyncio
import asyncio.events
import collections
import contextlib
import gc
import logging
import os
import pprint
import re
import select
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
import uvloop
class SSLTestCase:

    def _create_server_ssl_context(self, certfile, keyfile=None):
        if hasattr(ssl, 'PROTOCOL_TLS'):
            sslcontext = ssl.SSLContext(ssl.PROTOCOL_TLS)
        else:
            sslcontext = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
        sslcontext.options |= ssl.OP_NO_SSLv2
        sslcontext.load_cert_chain(certfile, keyfile)
        return sslcontext

    def _create_client_ssl_context(self, *, disable_verify=True):
        sslcontext = ssl.create_default_context()
        sslcontext.check_hostname = False
        if disable_verify:
            sslcontext.verify_mode = ssl.CERT_NONE
        return sslcontext

    @contextlib.contextmanager
    def _silence_eof_received_warning(self):
        logger = logging.getLogger('asyncio')
        filter = logging.Filter('has no effect when using ssl')
        logger.addFilter(filter)
        try:
            yield
        finally:
            logger.removeFilter(filter)