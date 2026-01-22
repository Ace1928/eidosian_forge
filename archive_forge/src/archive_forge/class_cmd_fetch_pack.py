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
class cmd_fetch_pack(Command):

    def run(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--all', action='store_true')
        parser.add_argument('location', nargs='?', type=str)
        args = parser.parse_args(argv)
        client, path = get_transport_and_path(args.location)
        r = Repo('.')
        if args.all:
            determine_wants = r.object_store.determine_wants_all
        else:

            def determine_wants(x, **kwargs):
                return [y for y in args if y not in r.object_store]
        client.fetch(path, r, determine_wants)