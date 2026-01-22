from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
class NicerConfigParser(ConfigParser):

    def __init__(self, filename, *args, **kw):
        ConfigParser.__init__(self, *args, **kw)
        self.filename = filename
        self._interpolation = self.InterpolateWrapper(self._interpolation)

    def defaults(self):
        """Return the defaults, with their values interpolated (with the
        defaults dict itself)

        Mainly to support defaults using values such as %(here)s
        """
        defaults = ConfigParser.defaults(self).copy()
        for key, val in defaults.items():
            defaults[key] = self.get('DEFAULT', key) or val
        return defaults

    class InterpolateWrapper:

        def __init__(self, original):
            self._original = original

        def __getattr__(self, name):
            return getattr(self._original, name)

        def before_get(self, parser, section, option, value, defaults):
            try:
                return self._original.before_get(parser, section, option, value, defaults)
            except Exception:
                e = sys.exc_info()[1]
                args = list(e.args)
                args[0] = f'Error in file {parser.filename}: {e}'
                e.args = tuple(args)
                e.message = args[0]
                raise