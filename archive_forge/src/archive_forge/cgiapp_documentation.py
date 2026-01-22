import os
import sys
import subprocess
from urllib.parse import quote
from paste.util import converters

    Paste Deploy interface for :class:`CGIApplication`

    This object acts as a proxy to a CGI application.  You pass in the
    script path (``script``), an optional path to search for the
    script (if the name isn't absolute) (``path``).  If you don't give
    a path, then ``$PATH`` will be used.
    