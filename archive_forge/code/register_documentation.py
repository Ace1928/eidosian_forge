import getpass
import io
import urllib.parse, urllib.request
from warnings import warn
from distutils.core import PyPIRCCommand
from distutils.errors import *
from distutils import log
 Post a query to the server, and return a string response.
        