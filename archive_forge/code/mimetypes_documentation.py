import os
import sys
import posixpath
import urllib.parse

        Load the MIME types database from Windows registry.

        If strict is true, information will be added to
        list of standard types, else to the list of non-standard
        types.
        