from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def write_xref_and_trailer(self, new_root_ref=None):
    if new_root_ref:
        self.del_root()
        self.root_ref = new_root_ref
    if self.info:
        self.info_ref = self.write_obj(None, self.info)
    start_xref = self.xref_table.write(self.f)
    num_entries = len(self.xref_table)
    trailer_dict = {b'Root': self.root_ref, b'Size': num_entries}
    if self.last_xref_section_offset is not None:
        trailer_dict[b'Prev'] = self.last_xref_section_offset
    if self.info:
        trailer_dict[b'Info'] = self.info_ref
    self.last_xref_section_offset = start_xref
    self.f.write(b'trailer\n' + bytes(PdfDict(trailer_dict)) + b'\nstartxref\n%d\n%%%%EOF' % start_xref)