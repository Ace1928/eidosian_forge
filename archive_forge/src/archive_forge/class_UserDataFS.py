import typing
import abc
import six
from appdirs import AppDirs
from ._repr import make_repr
from .osfs import OSFS
class UserDataFS(_AppFS):
    """A filesystem for per-user application data.

    May also be opened with
    ``open_fs('userdata://appname:author:version')``.

    """
    app_dir = 'user_data_dir'