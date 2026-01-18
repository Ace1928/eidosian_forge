from lxml import etree
import urllib
from .search import SearchManager
from .users import Users
from .resources import Project
from .tags import Tags
from .jsonutil import JsonTable
def unregister_callback(self):
    """ Unregisters the callback.
        """
    self._intf._callback = None