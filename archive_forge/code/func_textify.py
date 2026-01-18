import os
import sys
import re
from xml.sax.handler import ContentHandler
from ncclient.transport.errors import NetconfFramingError
from ncclient.transport.session import NetconfBase
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.operations.errors import OperationError
from ncclient.transport import SessionListener
import logging
def textify(buf):
    return buf.decode('UTF-8')