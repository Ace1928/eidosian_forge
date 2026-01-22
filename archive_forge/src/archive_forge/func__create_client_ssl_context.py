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
def _create_client_ssl_context(self, *, disable_verify=True):
    sslcontext = ssl.create_default_context()
    sslcontext.check_hostname = False
    if disable_verify:
        sslcontext.verify_mode = ssl.CERT_NONE
    return sslcontext