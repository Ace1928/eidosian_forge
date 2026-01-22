import os
from pathlib import Path
import stat
from itertools import islice, chain
from typing import Iterable, Optional, List, TextIO
from .translations import _
from .filelock import FileLock
def append_reload_and_write(self, s: str, filename: Path, encoding: str) -> None:
    if not self.hist_size:
        return self.append(s)
    try:
        fd = os.open(filename, os.O_APPEND | os.O_RDWR | os.O_CREAT, stat.S_IRUSR | stat.S_IWUSR)
        with open(fd, 'a+', encoding=encoding, errors='ignore') as hfile:
            with FileLock(hfile, filename=str(filename)):
                hfile.seek(0, os.SEEK_SET)
                entries = self.load_from(hfile)
                self.append_to(entries, s)
                hfile.seek(0, os.SEEK_SET)
                hfile.truncate()
                self.save_to(hfile, entries, self.hist_size)
                self.entries = entries
    except OSError as err:
        raise RuntimeError(_('Error occurred while writing to file %s (%s)') % (filename, err.strerror))
    else:
        if len(self.entries) == 0:
            self.entries = ['']