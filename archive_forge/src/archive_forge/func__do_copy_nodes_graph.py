from .. import errors
from .. import transport as _mod_transport
from ..lazy_import import lazy_import
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.knit import (
from ..bzr import btree_index
from ..bzr.index import (CombinedGraphIndex, GraphIndex,
from ..bzr.vf_repository import StreamSource
from .knitrepo import KnitRepository
from .pack_repo import (NewPack, PackCommitBuilder, Packer, PackRepository,
def _do_copy_nodes_graph(self, index_map, writer, write_index, output_lines, pb, readv_group_iter, total_items):
    knit = KnitVersionedFiles(None, None)
    if output_lines:
        factory = KnitPlainFactory()
    record_index = 0
    pb.update('Copied record', record_index, total_items)
    for index, readv_vector, node_vector in readv_group_iter:
        pack_obj = index_map[index]
        transport, path = pack_obj.access_tuple()
        try:
            reader = pack.make_readv_reader(transport, path, readv_vector)
        except _mod_transport.NoSuchFile:
            if self._reload_func is not None:
                self._reload_func()
            raise
        for (names, read_func), (key, eol_flag, references) in zip(reader.iter_records(), node_vector):
            raw_data = read_func(None)
            if output_lines:
                content, _ = knit._parse_record(key[-1], raw_data)
                if len(references[-1]) == 0:
                    line_iterator = factory.get_fulltext_content(content)
                else:
                    line_iterator = factory.get_linedelta_content(content)
                for line in line_iterator:
                    yield (line, key)
            else:
                df, _ = knit._parse_record_header(key, raw_data)
                df.close()
            pos, size = writer.add_bytes_record([raw_data], len(raw_data), names)
            write_index.add_node(key, eol_flag + b'%d %d' % (pos, size), references)
            pb.update('Copied record', record_index)
            record_index += 1