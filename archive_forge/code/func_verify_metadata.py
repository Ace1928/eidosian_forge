import getpass
import io
import urllib.parse, urllib.request
from warnings import warn
from distutils.core import PyPIRCCommand
from distutils.errors import *
from distutils import log
def verify_metadata(self):
    """ Send the metadata to the package index server to be checked.
        """
    code, result = self.post_to_server(self.build_post_data('verify'))
    log.info('Server response (%s): %s', code, result)