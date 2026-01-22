import os
import re
import sys
import time
from io import BytesIO
from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import (
from dulwich import log_utils
from .protocol import ReceivableProtocol
from .repo import BaseRepo, NotGitRepository, Repo
from .server import (
class WSGIServerLogger(WSGIServer):

    def handle_error(self, request, client_address):
        """Handle an error."""
        logger.exception('Exception happened during processing of request from %s' % str(client_address))