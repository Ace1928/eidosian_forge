import io
import os
import re
import sys
import time
import socket
import base64
import tempfile
import logging
from pyomo.common.dependencies import attempt_import
class NEOS(object):
    scheme = 'https'
    host = 'neos-server.org'
    port = '3333'