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
class cmd_log(Command):

    def run(self, args):
        parser = optparse.OptionParser()
        parser.add_option('--reverse', dest='reverse', action='store_true', help='Reverse order in which entries are printed')
        parser.add_option('--name-status', dest='name_status', action='store_true', help='Print name/status for each changed file')
        options, args = parser.parse_args(args)
        porcelain.log('.', paths=args, reverse=options.reverse, name_status=options.name_status, outstream=sys.stdout)