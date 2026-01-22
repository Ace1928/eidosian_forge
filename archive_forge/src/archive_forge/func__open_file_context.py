from contextlib import contextmanager
from ._miobase import _get_matfile_version, docfiller
from ._mio4 import MatFile4Reader, MatFile4Writer
from ._mio5 import MatFile5Reader, MatFile5Writer
@contextmanager
def _open_file_context(file_like, appendmat, mode='rb'):
    f, opened = _open_file(file_like, appendmat, mode)
    try:
        yield f
    finally:
        if opened:
            f.close()