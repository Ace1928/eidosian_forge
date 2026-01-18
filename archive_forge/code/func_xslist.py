from suds import *
from suds.reader import DocumentReader
from suds.sax import Namespace
from suds.transport import TransportError
from suds.xsd import *
from suds.xsd.query import *
from suds.xsd.sxbase import *
from urllib.parse import urljoin
from logging import getLogger
def xslist(self):
    return True