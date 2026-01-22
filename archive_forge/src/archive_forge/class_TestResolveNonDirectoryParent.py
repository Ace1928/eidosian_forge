import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
class TestResolveNonDirectoryParent(TestResolveConflicts):
    preamble = '\n$ brz init trunk\n...\n$ cd trunk\n$ brz mkdir foo\n...\n$ brz commit -m \'Create trunk\' -q\n$ echo "Boing" >foo/bar\n$ brz add -q foo/bar\n$ brz commit -q -m \'Add foo/bar\'\n$ brz branch -q . -r 1 ../branch\n$ cd ../branch\n$ rm -r foo\n$ echo "Boo!" >foo\n$ brz commit -q -m \'foo is now a file\'\n$ brz merge ../trunk\n2>RK  foo => foo.new/\n2>+N  foo.new/bar\n# FIXME: The message is misleading, foo.new *is* a directory when the message\n# is displayed -- vila 090916\n2>Conflict: foo.new is not a directory, but has files in it.  Created directory.\n2>1 conflicts encountered.\n'

    def test_take_this(self):
        self.run_script("\n$ brz rm -q foo.new --no-backup\n# FIXME: Isn't it weird that foo is now unkown even if foo.new has been put\n# aside ? -- vila 090916\n$ brz add -q foo\n$ brz resolve foo.new\n2>1 conflict resolved, 0 remaining\n$ brz commit -q --strict -m 'No more conflicts nor unknown files'\n")

    def test_take_other(self):
        self.run_script("\n$ brz rm -q foo --no-backup\n$ brz mv -q foo.new foo\n$ brz resolve foo\n2>1 conflict resolved, 0 remaining\n$ brz commit -q --strict -m 'No more conflicts nor unknown files'\n")

    def test_resolve_taking_this(self):
        self.run_script("\n$ brz resolve --take-this foo.new\n2>...\n$ brz commit -q --strict -m 'No more conflicts nor unknown files'\n")

    def test_resolve_taking_other(self):
        self.run_script("\n$ brz resolve --take-other foo.new\n2>...\n$ brz commit -q --strict -m 'No more conflicts nor unknown files'\n")