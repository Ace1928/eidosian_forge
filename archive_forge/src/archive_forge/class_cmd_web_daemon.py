import argparse
import optparse
import os
import signal
import sys
from getopt import getopt
from typing import ClassVar, Dict, Optional, Type
from dulwich import porcelain
from .client import GitProtocolError, get_transport_and_path
from .errors import ApplyDeltaError
from .index import Index
from .objectspec import parse_commit
from .pack import Pack, sha_to_hex
from .repo import Repo
class cmd_web_daemon(Command):

    def run(self, args):
        from dulwich import log_utils
        parser = optparse.OptionParser()
        parser.add_option('-l', '--listen_address', dest='listen_address', default='', help='Binding IP address.')
        parser.add_option('-p', '--port', dest='port', type=int, default=8000, help='Binding TCP port.')
        options, args = parser.parse_args(args)
        log_utils.default_logging_config()
        if len(args) >= 1:
            gitdir = args[0]
        else:
            gitdir = '.'
        porcelain.web_daemon(gitdir, address=options.listen_address, port=options.port)