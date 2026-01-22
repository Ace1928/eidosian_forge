import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def _get_sh_special_folder_path(csidl):
    """Call SHGetSpecialFolderPathW if available, or return None.

    Result is always unicode (or None).
    """
    try:
        SHGetSpecialFolderPath = ctypes.windll.shell32.SHGetSpecialFolderPathW
    except AttributeError:
        pass
    else:
        buf = ctypes.create_unicode_buffer(MAX_PATH)
        if SHGetSpecialFolderPath(None, buf, csidl, 0):
            return buf.value