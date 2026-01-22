import typing
import abc
import six
from appdirs import AppDirs
from ._repr import make_repr
from .osfs import OSFS
class SiteDataFS(_AppFS):
    """A filesystem for application site data.

    May also be opened with
    ``open_fs('sitedata://appname:author:version')``.

    """
    app_dir = 'site_data_dir'