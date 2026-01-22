from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class SeqXmlIterator(SequenceIterator):
    """Parser for seqXML files.

    Parses seqXML files and creates SeqRecords.
    Assumes valid seqXML please validate beforehand.
    It is assumed that all information for one record can be found within a
    record element or above. Two types of methods are called when the start
    tag of an element is reached. To receive only the attributes of an
    element before its end tag is reached implement _attr_TAGNAME.
    To get an element and its children as a DOM tree implement _elem_TAGNAME.
    Everything that is part of the DOM tree will not trigger any further
    method calls.
    """
    BLOCK = 1024

    def __init__(self, stream_or_path, namespace=None):
        """Create the object and initialize the XML parser."""
        self.parser = sax.make_parser()
        content_handler = ContentHandler()
        self.parser.setContentHandler(content_handler)
        self.parser.setFeature(handler.feature_namespaces, True)
        super().__init__(stream_or_path, mode='b', fmt='SeqXML')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        parser = self.parser
        content_handler = parser.getContentHandler()
        BLOCK = self.BLOCK
        while True:
            text = handle.read(BLOCK)
            if not text:
                if content_handler.startElementNS is None:
                    raise ValueError('Empty file.')
                else:
                    raise ValueError('XML file contains no data.')
            parser.feed(text)
            seqXMLversion = content_handler.seqXMLversion
            if seqXMLversion is not None:
                break
        self.seqXMLversion = seqXMLversion
        self.source = content_handler.source
        self.sourceVersion = content_handler.sourceVersion
        self.ncbiTaxID = content_handler.ncbiTaxID
        self.speciesName = content_handler.speciesName
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        """Iterate over the records in the XML file."""
        parser = self.parser
        content_handler = parser.getContentHandler()
        records = content_handler.records
        BLOCK = self.BLOCK
        while True:
            if len(records) > 1:
                record = records.pop(0)
                yield record
            text = handle.read(BLOCK)
            if not text:
                break
            parser.feed(text)
        yield from records
        records.clear()
        parser.close()