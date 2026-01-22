from typing import TYPE_CHECKING
from .. import config, controldir, errors, pyutils, registry
from .. import transport as _mod_transport
from ..branch import format_registry as branch_format_registry
from ..repository import format_registry as repository_format_registry
from ..workingtree import format_registry as workingtree_format_registry
class LineEndingError(errors.BzrError):
    _fmt = 'Line ending corrupted for file: %(file)s; Maybe your files got corrupted in transport?'

    def __init__(self, file):
        self.file = file