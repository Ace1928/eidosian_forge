import sys as _sys
from xml.sax import make_parser, handler
from netaddr.core import Publisher, Subscriber
from netaddr.ip import IPAddress, IPNetwork, IPRange, cidr_abbrev_to_verbose
from netaddr.compat import _open_binary
class IPv4Parser(XMLRecordParser):
    """
    A XMLRecordParser that understands how to parse and retrieve data records
    from the IANA IPv4 address space file.

    It can be found online here :-

        - http://www.iana.org/assignments/ipv4-address-space/ipv4-address-space.xml
    """

    def __init__(self, fh, **kwargs):
        """
        Constructor.

        fh - a valid, open file handle to an IANA IPv4 address space file.

        kwargs - additional parser options.
        """
        super(IPv4Parser, self).__init__(fh)

    def process_record(self, rec):
        """
        Callback method invoked for every record.

        See base class method for more details.
        """
        record = {}
        for key in ('prefix', 'designation', 'date', 'whois', 'status'):
            record[key] = str(rec.get(key, '')).strip()
        if '/' in record['prefix']:
            octet, prefix = record['prefix'].split('/')
            record['prefix'] = '%d/%d' % (int(octet), int(prefix))
        record['status'] = record['status'].capitalize()
        return record