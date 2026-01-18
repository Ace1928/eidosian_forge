import email.parser
import time
from difflib import SequenceMatcher
from typing import BinaryIO, Optional, TextIO, Union
from .objects import S_ISGITLINK, Blob, Commit
from .pack import ObjectContainer
def write_commit_patch(f, commit, contents, progress, version=None, encoding=None):
    """Write a individual file patch.

    Args:
      commit: Commit object
      progress: Tuple with current patch number and total.

    Returns:
      tuple with filename and contents
    """
    encoding = encoding or getattr(f, 'encoding', 'ascii')
    if isinstance(contents, str):
        contents = contents.encode(encoding)
    num, total = progress
    f.write(b'From ' + commit.id + b' ' + time.ctime(commit.commit_time).encode(encoding) + b'\n')
    f.write(b'From: ' + commit.author + b'\n')
    f.write(b'Date: ' + time.strftime('%a, %d %b %Y %H:%M:%S %Z').encode(encoding) + b'\n')
    f.write(('Subject: [PATCH %d/%d] ' % (num, total)).encode(encoding) + commit.message + b'\n')
    f.write(b'\n')
    f.write(b'---\n')
    try:
        import subprocess
        p = subprocess.Popen(['diffstat'], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    except (ImportError, OSError):
        pass
    else:
        diffstat, _ = p.communicate(contents)
        f.write(diffstat)
        f.write(b'\n')
    f.write(contents)
    f.write(b'-- \n')
    if version is None:
        from dulwich import __version__ as dulwich_version
        f.write(b'Dulwich %d.%d.%d\n' % dulwich_version)
    else:
        f.write(version.encode(encoding) + b'\n')