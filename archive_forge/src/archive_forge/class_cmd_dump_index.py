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
class cmd_dump_index(Command):

    def run(self, args):
        opts, args = getopt(args, '', [])
        if args == []:
            print('Usage: dulwich dump-index FILENAME')
            sys.exit(1)
        filename = args[0]
        idx = Index(filename)
        for o in idx:
            print(o, idx[o])