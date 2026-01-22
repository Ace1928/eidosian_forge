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
class cmd_rev_list(Command):

    def run(self, args):
        opts, args = getopt(args, '', [])
        if len(args) < 1:
            print('Usage: dulwich rev-list COMMITID...')
            sys.exit(1)
        porcelain.rev_list('.', args)