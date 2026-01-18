import base64
import ctypes
import itertools
import os
import re
import ssl
import struct
import tempfile
from .bindings import CFConst, CoreFoundation, Security

    Builds a TLS alert record for an unknown CA.
    