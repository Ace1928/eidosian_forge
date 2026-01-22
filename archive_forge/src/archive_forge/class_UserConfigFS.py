import typing
import abc
import six
from appdirs import AppDirs
from ._repr import make_repr
from .osfs import OSFS
class UserConfigFS(_AppFS):
    """A filesystem for per-user config data.

    May also be opened with
    ``open_fs('userconf://appname:author:version')``.

    """
    app_dir = 'user_config_dir'