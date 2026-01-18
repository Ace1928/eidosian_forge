import os
import sys
import threading
from dulwich.tests import skipIf
from ...server import DictBackend, TCPGitServer
from .server_utils import NoSideBand64kReceivePackHandler, ServerTests
from .utils import CompatTestCase, require_git_version
Tests for client/server compatibility with side-band-64k support.