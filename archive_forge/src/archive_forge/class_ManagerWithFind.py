import abc
import copy
from http import client as http_client
from urllib import parse as urlparse
from oslo_utils import strutils
from ironicclient.common.apiclient import exceptions
from ironicclient.common.i18n import _
class ManagerWithFind(BaseManager, metaclass=abc.ABCMeta):
    """Manager with additional `find()`/`findall()` methods."""

    @abc.abstractmethod
    def list(self):
        pass

    def find(self, **kwargs):
        """Find a single item with attributes matching ``**kwargs``.

        This isn't very efficient: it loads the entire list then filters on
        the Python side.
        """
        matches = self.findall(**kwargs)
        num_matches = len(matches)
        if num_matches == 0:
            msg = _('No %(name)s matching %(args)s.') % {'name': self.resource_class.__name__, 'args': kwargs}
            raise exceptions.NotFound(msg)
        elif num_matches > 1:
            raise exceptions.NoUniqueMatch()
        else:
            return matches[0]

    def findall(self, **kwargs):
        """Find all items with attributes matching ``**kwargs``.

        This isn't very efficient: it loads the entire list then filters on
        the Python side.
        """
        found = []
        searches = kwargs.items()
        for obj in self.list():
            try:
                if all((getattr(obj, attr) == value for attr, value in searches)):
                    found.append(obj)
            except AttributeError:
                continue
        return found