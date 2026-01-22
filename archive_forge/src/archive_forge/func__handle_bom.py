import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def _handle_bom(self, infile):
    """
        Handle any BOM, and decode if necessary.
        
        If an encoding is specified, that *must* be used - but the BOM should
        still be removed (and the BOM attribute set).
        
        (If the encoding is wrongly specified, then a BOM for an alternative
        encoding won't be discovered or removed.)
        
        If an encoding is not specified, UTF8 or UTF16 BOM will be detected and
        removed. The BOM attribute will be set. UTF16 will be decoded to
        unicode.
        
        NOTE: This method must not be called with an empty ``infile``.
        
        Specifying the *wrong* encoding is likely to cause a
        ``UnicodeDecodeError``.
        
        ``infile`` must always be returned as a list of lines, but may be
        passed in as a single string.
        """
    if self.encoding is not None and self.encoding.lower() not in BOM_LIST:
        return self._decode(infile, self.encoding)
    if isinstance(infile, (list, tuple)):
        line = infile[0]
    else:
        line = infile
    if isinstance(line, six.text_type):
        return self._decode(infile, self.encoding)
    if self.encoding is not None:
        enc = BOM_LIST[self.encoding.lower()]
        if enc == 'utf_16':
            for BOM, (encoding, final_encoding) in list(BOMS.items()):
                if not final_encoding:
                    continue
                if infile.startswith(BOM):
                    return self._decode(infile, encoding)
            return self._decode(infile, self.encoding)
        BOM = BOM_SET[enc]
        if not line.startswith(BOM):
            return self._decode(infile, self.encoding)
        newline = line[len(BOM):]
        if isinstance(infile, (list, tuple)):
            infile[0] = newline
        else:
            infile = newline
        self.BOM = True
        return self._decode(infile, self.encoding)
    for BOM, (encoding, final_encoding) in list(BOMS.items()):
        if not isinstance(line, six.binary_type) or not line.startswith(BOM):
            continue
        else:
            self.encoding = final_encoding
            if not final_encoding:
                self.BOM = True
                newline = line[len(BOM):]
                if isinstance(infile, (list, tuple)):
                    infile[0] = newline
                else:
                    infile = newline
                if isinstance(infile, six.text_type):
                    return infile.splitlines(True)
                elif isinstance(infile, six.binary_type):
                    return infile.decode('utf-8').splitlines(True)
                else:
                    return self._decode(infile, 'utf-8')
            return self._decode(infile, encoding)
    if six.PY2 and isinstance(line, str):
        return self._decode(infile, None)
    if isinstance(infile, six.binary_type):
        return infile.decode('utf-8').splitlines(True)
    else:
        return self._decode(infile, 'utf-8')