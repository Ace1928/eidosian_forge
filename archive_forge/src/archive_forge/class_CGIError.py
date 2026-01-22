import os
import sys
import subprocess
from urllib.parse import quote
from paste.util import converters
class CGIError(Exception):
    """
    Raised when the CGI script can't be found or doesn't
    act like a proper CGI script.
    """