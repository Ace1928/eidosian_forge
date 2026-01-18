from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def write_catalog(self):
    self.del_root()
    self.root_ref = self.next_object_id(self.f.tell())
    self.pages_ref = self.next_object_id(0)
    self.rewrite_pages()
    self.write_obj(self.root_ref, Type=PdfName(b'Catalog'), Pages=self.pages_ref)
    self.write_obj(self.pages_ref, Type=PdfName(b'Pages'), Count=len(self.pages), Kids=self.pages)
    return self.root_ref