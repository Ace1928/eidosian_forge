import sys as _sys
from xml.sax import make_parser, handler
from netaddr.core import Publisher, Subscriber
from netaddr.ip import IPAddress, IPNetwork, IPRange, cidr_abbrev_to_verbose
from netaddr.compat import _open_binary
class XMLRecordParser(Publisher):
    """
    A configurable Parser that understands how to parse XML based records.
    """

    def __init__(self, fh, **kwargs):
        """
        Constructor.

        fh - a valid, open file handle to XML based record data.
        """
        super(XMLRecordParser, self).__init__()
        self.xmlparser = make_parser()
        self.xmlparser.setContentHandler(SaxRecordParser(self.consume_record))
        self.fh = fh
        self.__dict__.update(kwargs)

    def process_record(self, rec):
        """
        This is the callback method invoked for every record. It is usually
        over-ridden by base classes to provide specific record-based logic.

        Any record can be vetoed (not passed to registered Subscriber objects)
        by simply returning None.
        """
        return rec

    def consume_record(self, rec):
        record = self.process_record(rec)
        if record is not None:
            self.notify(record)

    def parse(self):
        """
        Parse and normalises records, notifying registered subscribers with
        record data as it is encountered.
        """
        self.xmlparser.parse(self.fh)