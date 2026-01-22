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
class cmd_commit_tree(Command):

    def run(self, args):
        opts, args = getopt(args, '', ['message'])
        if args == []:
            print('usage: dulwich commit-tree tree')
            sys.exit(1)
        opts = dict(opts)
        porcelain.commit_tree('.', tree=args[0], message=opts['--message'])