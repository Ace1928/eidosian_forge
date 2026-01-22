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
class cmd_archive(Command):

    def run(self, args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--remote', type=str, help='Retrieve archive from specified remote repo')
        parser.add_argument('committish', type=str, nargs='?')
        args = parser.parse_args(args)
        if args.remote:
            client, path = get_transport_and_path(args.remote)
            client.archive(path, args.committish, sys.stdout.write, write_error=sys.stderr.write)
        else:
            porcelain.archive('.', args.committish, outstream=sys.stdout.buffer, errstream=sys.stderr)