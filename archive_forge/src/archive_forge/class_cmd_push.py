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
class cmd_push(Command):

    def run(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', '--force', action='store_true', help='Force')
        parser.add_argument('to_location', type=str)
        parser.add_argument('refspec', type=str, nargs='*')
        args = parser.parse_args(argv)
        try:
            porcelain.push('.', args.to_location, args.refspec or None, force=args.force)
        except porcelain.DivergedBranches:
            sys.stderr.write('Diverged branches; specify --force to override')
            return 1