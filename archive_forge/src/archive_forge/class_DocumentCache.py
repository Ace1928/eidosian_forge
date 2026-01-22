import suds
import suds.sax.element
import suds.sax.parser
import datetime
import os
import shutil
import tempfile
from logging import getLogger
class DocumentCache(FileCache):
    """XML document file cache."""

    def fnsuffix(self):
        return 'xml'

    def get(self, id):
        fp = None
        try:
            fp = self._getf(id)
            if fp is not None:
                p = suds.sax.parser.Parser()
                cached = p.parse(fp)
                fp.close()
                return cached
        except Exception:
            if fp is not None:
                fp.close()
            self.purge(id)

    def put(self, id, object):
        if isinstance(object, (suds.sax.document.Document, suds.sax.element.Element)):
            super(DocumentCache, self).put(id, suds.byte_str(str(object)))
        return object