from __future__ import print_function, absolute_import
import threading
import warnings
from lxml import etree as _etree
from .common import DTDForbidden, EntitiesForbidden, NotSupportedError
Check docinfo of an element tree for DTD and entity declarations

    The check for entity declarations needs lxml 3 or newer. lxml 2.x does
    not support dtd.iterentities().
    