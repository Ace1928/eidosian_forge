import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
This works identically to NNTP.__init__, except for the change
            in default port and the `ssl_context` argument for SSL connections.
            