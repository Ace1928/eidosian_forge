from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
class IDNA2008Codec(IDNACodec):
    """IDNA 2008 encoder/decoder.

        *uts_46* is a ``bool``.  If True, apply Unicode IDNA
        compatibility processing as described in Unicode Technical
        Standard #46 (http://unicode.org/reports/tr46/).
        If False, do not apply the mapping.  The default is False.

        *transitional* is a ``bool``: If True, use the
        "transitional" mode described in Unicode Technical Standard
        #46.  The default is False.

        *allow_pure_ascii* is a ``bool``.  If True, then a label which
        consists of only ASCII characters is allowed.  This is less
        strict than regular IDNA 2008, but is also necessary for mixed
        names, e.g. a name with starting with "_sip._tcp." and ending
        in an IDN suffix which would otherwise be disallowed.  The
        default is False.

        *strict_decode* is a ``bool``: If True, then IDNA2008 checking
        is done when decoding.  This can cause failures if the name
        was encoded with IDNA2003.  The default is False.
        """

    def __init__(self, uts_46=False, transitional=False, allow_pure_ascii=False, strict_decode=False):
        """Initialize the IDNA 2008 encoder/decoder."""
        super(IDNA2008Codec, self).__init__()
        self.uts_46 = uts_46
        self.transitional = transitional
        self.allow_pure_ascii = allow_pure_ascii
        self.strict_decode = strict_decode

    def is_all_ascii(self, label):
        for c in label:
            if ord(c) > 127:
                return False
        return True

    def encode(self, label):
        if label == '':
            return b''
        if self.allow_pure_ascii and self.is_all_ascii(label):
            return label.encode('ascii')
        if not have_idna_2008:
            raise NoIDNA2008
        try:
            if self.uts_46:
                label = idna.uts46_remap(label, False, self.transitional)
            return idna.alabel(label)
        except idna.IDNAError as e:
            raise IDNAException(idna_exception=e)

    def decode(self, label):
        if not self.strict_decode:
            return super(IDNA2008Codec, self).decode(label)
        if label == b'':
            return u''
        if not have_idna_2008:
            raise NoIDNA2008
        try:
            if self.uts_46:
                label = idna.uts46_remap(label, False, False)
            return _escapify(idna.ulabel(label), True)
        except idna.IDNAError as e:
            raise IDNAException(idna_exception=e)