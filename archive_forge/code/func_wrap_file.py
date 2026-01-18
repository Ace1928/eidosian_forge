import os
import mimetypes
from datetime import datetime
from time import gmtime
def wrap_file(environ, file, buffer_size=8192):
    """Wraps a file.  This uses the WSGI server's file wrapper if available
    or otherwise the generic :class:`FileWrapper`.

    If the file wrapper from the WSGI server is used it's important to not
    iterate over it from inside the application but to pass it through
    unchanged.

    More information about file wrappers are available in :pep:`333`.

    :param file: a :class:`file`-like object with a :meth:`~file.read` method.
    :param buffer_size: number of bytes for one iteration.
    """
    return environ.get('wsgi.file_wrapper', FileWrapper)(file, buffer_size)