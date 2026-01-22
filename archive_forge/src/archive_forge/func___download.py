from suds import *
from suds.reader import DocumentReader
from suds.sax import Namespace
from suds.transport import TransportError
from suds.xsd import *
from suds.xsd.query import *
from suds.xsd.sxbase import *
from urllib.parse import urljoin
from logging import getLogger
def __download(self, url, loaded_schemata, options):
    """Download the schema."""
    try:
        reader = DocumentReader(options)
        d = reader.open(url)
        root = d.root()
        root.set('url', url)
        self.__applytns(root)
        return self.schema.instance(root, url, loaded_schemata, options)
    except TransportError:
        msg = 'include schema at (%s), failed' % url
        log.error('%s, %s', self.id, msg, exc_info=True)
        raise Exception(msg)