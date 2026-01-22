import os
import tempfile
from contextlib import contextmanager
from .deprecated import deprecate_with_version
@contextmanager
def InTemporaryDirectory():
    """Create, return, and change directory to a temporary directory

    Notes
    -----
    As its name suggests, the class temporarily changes the working
    directory of the Python process, and this is not thread-safe.  We suggest
    using it only for tests.

    Examples
    --------
    >>> import os
    >>> from pathlib import Path
    >>> my_cwd = os.getcwd()
    >>> with InTemporaryDirectory() as tmpdir:
    ...     _ = Path('test.txt').write_text('some text')
    ...     assert os.path.isfile('test.txt')
    ...     assert os.path.isfile(os.path.join(tmpdir, 'test.txt'))
    >>> os.path.exists(tmpdir)
    False
    >>> os.getcwd() == my_cwd
    True
    """
    with tempfile.TemporaryDirectory() as tmpdir, _chdir(tmpdir):
        yield tmpdir