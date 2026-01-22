import suds
import suds.sax.element
import suds.sax.parser
import datetime
import os
import shutil
import tempfile
from logging import getLogger
def __open(self, filename, *args):
    """Open cache file making sure the I{location} folder is created."""
    self.__mktmp()
    return open(filename, *args)