import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
class PlugIn:
    """The breezy representation of a plugin.

    The PlugIn object provides a way to manipulate a given plugin module.
    """

    def __init__(self, name, module):
        """Construct a plugin for module."""
        self.name = name
        self.module = module

    def path(self):
        """Get the path that this plugin was loaded from."""
        if getattr(self.module, '__path__', None) is not None:
            return os.path.abspath(self.module.__path__[0])
        elif getattr(self.module, '__file__', None) is not None:
            path = os.path.abspath(self.module.__file__)
            if path[-4:] == COMPILED_EXT:
                pypath = path[:-4] + '.py'
                if os.path.isfile(pypath):
                    path = pypath
            return path
        else:
            return repr(self.module)

    def __repr__(self):
        return '<{}.{} name={}, module={}>'.format(self.__class__.__module__, self.__class__.__name__, self.name, self.module)

    def test_suite(self):
        """Return the plugin's test suite."""
        if getattr(self.module, 'test_suite', None) is not None:
            return self.module.test_suite()
        else:
            return None

    def load_plugin_tests(self, loader):
        """Return the adapted plugin's test suite.

        Args:
          loader: The custom loader that should be used to load additional
            tests.
        """
        if getattr(self.module, 'load_tests', None) is not None:
            return loader.loadTestsFromModule(self.module)
        else:
            return None

    def version_info(self):
        """Return the plugin's version_tuple or None if unknown."""
        version_info = getattr(self.module, 'version_info', None)
        if version_info is not None:
            try:
                if isinstance(version_info, str):
                    version_info = version_info.split('.')
                elif len(version_info) == 3:
                    version_info = tuple(version_info) + ('final', 0)
            except TypeError:
                trace.log_exception_quietly()
                version_info = (version_info,)
        return version_info

    @property
    def __version__(self):
        version_info = self.version_info()
        if version_info is None or len(version_info) == 0:
            return 'unknown'
        try:
            version_string = breezy._format_version_tuple(version_info)
        except (ValueError, TypeError, IndexError):
            trace.log_exception_quietly()
            version_string = '.'.join(map(str, version_info))
        return version_string