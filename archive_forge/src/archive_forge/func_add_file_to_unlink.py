import atexit
import os
import signal
import sys
import ovs.vlog
def add_file_to_unlink(file):
    """Registers 'file' to be unlinked when the program terminates via
    sys.exit() or a fatal signal."""
    global _added_hook
    if not _added_hook:
        _added_hook = True
        add_hook(_unlink_files, _cancel_files, True)
    _files[file] = None