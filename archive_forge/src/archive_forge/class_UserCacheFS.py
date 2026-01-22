import typing
import abc
import six
from appdirs import AppDirs
from ._repr import make_repr
from .osfs import OSFS
class UserCacheFS(_AppFS):
    """A filesystem for per-user application cache data.

    May also be opened with
    ``open_fs('usercache://appname:author:version')``.

    """
    app_dir = 'user_cache_dir'