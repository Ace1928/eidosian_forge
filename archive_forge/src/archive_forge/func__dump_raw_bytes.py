from io import BytesIO
from .. import errors, osutils, transport
from ..commands import Command, display_command
from ..option import Option
from ..workingtree import WorkingTree
from . import btree_index, static_tuple
def _dump_raw_bytes(self, trans, basename):
    import zlib
    bt, bytes = self._get_index_and_bytes(trans, basename)
    for page_idx, page_start in enumerate(range(0, len(bytes), btree_index._PAGE_SIZE)):
        page_end = min(page_start + btree_index._PAGE_SIZE, len(bytes))
        page_bytes = bytes[page_start:page_end]
        if page_idx == 0:
            self.outf.write('Root node:\n')
            header_end, data = bt._parse_header_from_bytes(page_bytes)
            self.outf.write(page_bytes[:header_end])
            page_bytes = data
        self.outf.write('\nPage %d\n' % (page_idx,))
        if len(page_bytes) == 0:
            self.outf.write('(empty)\n')
        else:
            decomp_bytes = zlib.decompress(page_bytes)
            self.outf.write(decomp_bytes)
            self.outf.write('\n')