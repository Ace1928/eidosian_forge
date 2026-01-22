import io
import linecache
import warnings
from collections import OrderedDict
from html import escape
from typing import (
from xml.sax import handler, make_parser
from xml.sax.xmlreader import AttributesNSImpl, Locator
from zope.interface import implementer
from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python import urlpath
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.reflect import fullyQualifiedName
from twisted.web import resource
from twisted.web._element import Element, renderer
from twisted.web._flatten import Flattenable, flatten, flattenString
from twisted.web._stan import CDATA, Comment, Tag, slot
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
@implementer(ITemplateLoader)
class XMLFile:
    """
    An L{ITemplateLoader} that loads and parses XML from a file.
    """

    def __init__(self, path: FilePath[Any]):
        """
        Run the parser on a file.

        @param path: The file from which to load the XML.
        """
        if not isinstance(path, FilePath):
            warnings.warn('Passing filenames or file objects to XMLFile is deprecated since Twisted 12.1.  Pass a FilePath instead.', category=DeprecationWarning, stacklevel=2)
        self._loadedTemplate: Optional[List['Flattenable']] = None
        'The loaded document, or L{None}, if not loaded.'
        self._path: FilePath[Any] = path
        'The file that is being loaded from.'

    def _loadDoc(self) -> List['Flattenable']:
        """
        Read and parse the XML.

        @return: the loaded document.
        """
        if not isinstance(self._path, FilePath):
            return _flatsaxParse(self._path)
        else:
            with self._path.open('r') as f:
                return _flatsaxParse(f)

    def __repr__(self) -> str:
        return f'<XMLFile of {self._path!r}>'

    def load(self) -> List['Flattenable']:
        """
        Return the document, first loading it if necessary.

        @return: the loaded document.
        """
        if self._loadedTemplate is None:
            self._loadedTemplate = self._loadDoc()
        return self._loadedTemplate