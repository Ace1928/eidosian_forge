import getpass
import io
import urllib.parse, urllib.request
from warnings import warn
from distutils.core import PyPIRCCommand
from distutils.errors import *
from distutils import log
def _set_config(self):
    """ Reads the configuration file and set attributes.
        """
    config = self._read_pypirc()
    if config != {}:
        self.username = config['username']
        self.password = config['password']
        self.repository = config['repository']
        self.realm = config['realm']
        self.has_config = True
    else:
        if self.repository not in ('pypi', self.DEFAULT_REPOSITORY):
            raise ValueError('%s not found in .pypirc' % self.repository)
        if self.repository == 'pypi':
            self.repository = self.DEFAULT_REPOSITORY
        self.has_config = False