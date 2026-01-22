import abc
from taskflow import exceptions as exc
from taskflow.persistence import base
from taskflow.persistence import models
class PathBasedBackend(base.Backend, metaclass=abc.ABCMeta):
    """Base class for persistence backends that address data by path

    Subclasses of this backend write logbooks, flow details, and atom details
    to a provided base path in some filesystem-like storage. They will create
    and store those objects in three key directories (one for logbooks, one
    for flow details and one for atom details). They create those associated
    directories and then create files inside those directories that represent
    the contents of those objects for later reading and writing.
    """
    DEFAULT_PATH = None

    def __init__(self, conf):
        super(PathBasedBackend, self).__init__(conf)
        self._path = self._conf.get('path', None)
        if not self._path:
            self._path = self.DEFAULT_PATH

    @property
    def path(self):
        return self._path