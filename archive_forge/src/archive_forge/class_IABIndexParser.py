import os.path as _path
import csv as _csv
from netaddr.compat import _open_binary
from netaddr.core import Subscriber, Publisher
class IABIndexParser(Publisher):
    """
    A concrete Publisher that parses IAB (Individual Address Block) records
    from IEEE text-based registration files

    It notifies registered Subscribers as each record is encountered, passing
    on the record's position relative to the start of the file (offset) and
    the size of the record (in bytes).

    The file processed by this parser is available online from this URL :-

        - http://standards.ieee.org/regauth/oui/iab.txt

    This is a sample of the record structure expected::

        00-50-C2   (hex)        ACME CORPORATION
        ABC000-ABCFFF     (base 16)        ACME CORPORATION
                        1 MAIN STREET
                        SPRINGFIELD
                        UNITED STATES
    """

    def __init__(self, ieee_file):
        """
        Constructor.

        :param ieee_file: a file-like object or name of file containing IAB
            records. When using a file-like object always open it in binary
            mode otherwise offsets will probably misbehave.
        """
        super(IABIndexParser, self).__init__()
        if hasattr(ieee_file, 'readline') and hasattr(ieee_file, 'tell'):
            self.fh = ieee_file
        else:
            self.fh = open(ieee_file, 'rb')

    def parse(self):
        """
        Starts the parsing process which detects records and notifies
        registered subscribers as it finds each IAB record.
        """
        skip_header = True
        record = None
        size = 0
        hex_marker = b'(hex)'
        base16_marker = b'(base 16)'
        hyphen = b'-'
        empty_string = b''
        while True:
            line = self.fh.readline()
            if not line:
                break
            if skip_header and hex_marker in line:
                skip_header = False
            if skip_header:
                continue
            if hex_marker in line:
                if record is not None:
                    record.append(size)
                    self.notify(record)
                offset = self.fh.tell() - len(line)
                iab_prefix = line.split()[0]
                index = iab_prefix
                record = [index, offset]
                size = len(line)
            elif base16_marker in line:
                size += len(line)
                prefix = record[0].replace(hyphen, empty_string)
                suffix = line.split()[0]
                suffix = suffix.split(hyphen)[0]
                record[0] = int(prefix + suffix, 16) >> 12
            else:
                size += len(line)
        record.append(size)
        self.notify(record)