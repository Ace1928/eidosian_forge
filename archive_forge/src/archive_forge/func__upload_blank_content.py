import itertools
from .. import errors, lockable_files, lockdir
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from ..repository import Repository, RepositoryFormat, format_registry
from . import bzrdir
def _upload_blank_content(self, a_bzrdir, dirs, files, utf8_files, shared):
    """Upload the initial blank content."""
    control_files = self._create_control_files(a_bzrdir)
    control_files.lock_write()
    transport = control_files._transport
    if shared is True:
        utf8_files += [('shared-storage', b'')]
    try:
        for dir in dirs:
            transport.mkdir(dir, mode=a_bzrdir._get_dir_mode())
        for filename, content_stream in files:
            transport.put_file(filename, content_stream, mode=a_bzrdir._get_file_mode())
        for filename, content_bytes in utf8_files:
            transport.put_bytes_non_atomic(filename, content_bytes, mode=a_bzrdir._get_file_mode())
    finally:
        control_files.unlock()