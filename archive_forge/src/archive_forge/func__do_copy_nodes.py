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
def _do_copy_nodes(self, nodes, index_map, writer, write_index, pb, output_lines=None):
    knit = KnitVersionedFiles(None, None)
    nodes = sorted(nodes)
    request_groups = {}
    for index, key, value in nodes:
        if index not in request_groups:
            request_groups[index] = []
        request_groups[index].append((key, value))
    record_index = 0
    pb.update('Copied record', record_index, len(nodes))
    for index, items in request_groups.items():
        pack_readv_requests = []
        for key, value in items:
            bits = value[1:].split(b' ')
            offset, length = (int(bits[0]), int(bits[1]))
            pack_readv_requests.append((offset, length, (key, value[0:1])))
        pack_readv_requests.sort()
        pack_obj = index_map[index]
        transport, path = pack_obj.access_tuple()
        try:
            reader = pack.make_readv_reader(transport, path, [offset[0:2] for offset in pack_readv_requests])
        except _mod_transport.NoSuchFile:
            if self._reload_func is not None:
                self._reload_func()
            raise
        for (names, read_func), (_1, _2, (key, eol_flag)) in zip(reader.iter_records(), pack_readv_requests):
            raw_data = read_func(None)
            if output_lines is not None:
                output_lines(knit._parse_record(key[-1], raw_data)[0])
            else:
                df, _ = knit._parse_record_header(key, raw_data)
                df.close()
            pos, size = writer.add_bytes_record([raw_data], len(raw_data), names)
            write_index.add_node(key, eol_flag + b'%d %d' % (pos, size))
            pb.update('Copied record', record_index)
            record_index += 1