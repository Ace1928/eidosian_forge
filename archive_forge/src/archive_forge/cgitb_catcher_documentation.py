import cgitb
from io import StringIO
import sys
from paste.util import converters

    Wraps the application in the ``cgitb`` (standard library)
    error catcher.

      display:
        If true (or debug is set in the global configuration)
        then the traceback will be displayed in the browser

      logdir:
        Writes logs of all errors in that directory

      context:
        Number of lines of context to show around each line of
        source code
    