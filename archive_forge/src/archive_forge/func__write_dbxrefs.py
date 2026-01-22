from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _write_dbxrefs(self, record):
    """Write all database cross references (PRIVATE)."""
    if record.dbxrefs is not None:
        for dbxref in record.dbxrefs:
            if not isinstance(dbxref, str):
                raise TypeError('dbxrefs should be of type list of string')
            if dbxref.find(':') < 1:
                raise ValueError("dbxrefs should be in the form ['source:id', 'source:id' ]")
            dbsource, dbid = dbxref.split(':', 1)
            attr = {'source': dbsource, 'id': dbid}
            self.xml_generator.startElement('DBRef', AttributesImpl(attr))
            self.xml_generator.endElement('DBRef')