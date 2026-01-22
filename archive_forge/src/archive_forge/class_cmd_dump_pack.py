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
class cmd_dump_pack(Command):

    def run(self, args):
        opts, args = getopt(args, '', [])
        if args == []:
            print('Usage: dulwich dump-pack FILENAME')
            sys.exit(1)
        basename, _ = os.path.splitext(args[0])
        x = Pack(basename)
        print('Object names checksum: %s' % x.name())
        print('Checksum: %s' % sha_to_hex(x.get_stored_checksum()))
        if not x.check():
            print('CHECKSUM DOES NOT MATCH')
        print('Length: %d' % len(x))
        for name in x:
            try:
                print('\t%s' % x[name])
            except KeyError as k:
                print(f'\t{name}: Unable to resolve base {k}')
            except ApplyDeltaError as e:
                print(f'\t{name}: Unable to apply delta: {e!r}')