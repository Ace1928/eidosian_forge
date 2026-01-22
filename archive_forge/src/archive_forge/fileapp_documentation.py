import os, time, mimetypes, zipfile, tarfile
from paste.httpexceptions import (
from paste.httpheaders import (

    Returns an application that serves up a DataApp for items requested
    in a given zip or tar archive.

    Constructor Arguments:

        ``filepath``    the path to the archive being served

    ``cache_control()``

        This method provides validated construction of the ``Cache-Control``
        header as well as providing for automated filling out of the
        ``EXPIRES`` header for HTTP/1.0 clients.
    