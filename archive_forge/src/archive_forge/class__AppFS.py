import typing
import abc
import six
from appdirs import AppDirs
from ._repr import make_repr
from .osfs import OSFS
@six.add_metaclass(_CopyInitMeta)
class _AppFS(OSFS):
    """Abstract base class for an app FS."""
    app_dir = None

    def __init__(self, appname, author=None, version=None, roaming=False, create=True):
        """Create a new application-specific filesystem.

        Arguments:
            appname (str): The name of the application.
            author (str): The name of the author (used on Windows).
            version (str): Optional version string, if a unique location
                per version of the application is required.
            roaming (bool): If `True`, use a *roaming* profile on
                Windows.
            create (bool): If `True` (the default) the directory
                will be created if it does not exist.

        """
        self.app_dirs = AppDirs(appname, author, version, roaming)
        self._create = create
        super(_AppFS, self).__init__(getattr(self.app_dirs, self.app_dir), create=create)

    def __repr__(self):
        return make_repr(self.__class__.__name__, self.app_dirs.appname, author=(self.app_dirs.appauthor, None), version=(self.app_dirs.version, None), roaming=(self.app_dirs.roaming, False), create=(self._create, True))

    def __str__(self):
        return "<{} '{}'>".format(self.__class__.__name__.lower(), self.app_dirs.appname)