import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def _handle_configspec(self, configspec):
    """Parse the configspec."""
    if not isinstance(configspec, ConfigObj):
        try:
            configspec = ConfigObj(configspec, raise_errors=True, file_error=True, _inspec=True)
        except ConfigObjError as e:
            raise ConfigspecError('Parsing configspec failed: %s' % e)
        except IOError as e:
            raise IOError('Reading configspec failed: %s' % e)
    self.configspec = configspec