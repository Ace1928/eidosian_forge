from io import BytesIO
from .. import errors, osutils, transport
from ..commands import Command, display_command
from ..option import Option
from ..workingtree import WorkingTree
from . import btree_index, static_tuple
class cmd_file_id(Command):
    __doc__ = 'Print file_id of a particular file or directory.\n\n    The file_id is assigned when the file is first added and remains the\n    same through all revisions where the file exists, even when it is\n    moved or renamed.\n    '
    hidden = True
    _see_also = ['inventory', 'ls']
    takes_args = ['filename']

    @display_command
    def run(self, filename):
        tree, relpath = WorkingTree.open_containing(filename)
        file_id = tree.path2id(relpath)
        if file_id is None:
            raise errors.NotVersionedError(filename)
        else:
            self.outf.write(file_id.decode('utf-8') + '\n')