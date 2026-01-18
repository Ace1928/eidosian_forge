from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom

        L{domhelpers.getNodeText} returns a C{unicode} string when text
        nodes are represented in the DOM with unicode, whether or not there
        are non-ASCII characters present.
        