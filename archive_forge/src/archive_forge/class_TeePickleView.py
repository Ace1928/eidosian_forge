from __future__ import absolute_import, print_function, division
from petl.compat import pickle, next
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
class TeePickleView(Table):

    def __init__(self, table, source=None, protocol=-1, write_header=True):
        self.table = table
        self.source = source
        self.protocol = protocol
        self.write_header = write_header

    def __iter__(self):
        protocol = self.protocol
        source = write_source_from_arg(self.source)
        with source.open('wb') as f:
            it = iter(self.table)
            try:
                hdr = next(it)
            except StopIteration:
                return
            if self.write_header:
                pickle.dump(hdr, f, protocol)
            yield tuple(hdr)
            for row in it:
                pickle.dump(row, f, protocol)
                yield tuple(row)