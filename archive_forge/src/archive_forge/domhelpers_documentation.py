import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
namedChildren(parent, nodeName) -> children (not descendants) of parent
    that have tagName == nodeName
    