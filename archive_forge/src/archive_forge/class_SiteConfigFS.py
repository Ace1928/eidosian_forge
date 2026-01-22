import typing
import abc
import six
from appdirs import AppDirs
from ._repr import make_repr
from .osfs import OSFS
class SiteConfigFS(_AppFS):
    """A filesystem for application config data.

    May also be opened with
    ``open_fs('siteconf://appname:author:version')``.

    """
    app_dir = 'site_config_dir'