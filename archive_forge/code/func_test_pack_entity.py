from gitdb.test.lib import (
from gitdb.stream import DeltaApplyReader
from gitdb.pack import (
from gitdb.base import (
from gitdb.fun import delta_types
from gitdb.exc import UnsupportedOperation
from gitdb.util import to_bin_sha
import pytest
import os
import tempfile
@with_rw_directory
def test_pack_entity(self, rw_dir):
    pack_objs = list()
    for packinfo, indexinfo in ((self.packfile_v2_1, self.packindexfile_v1), (self.packfile_v2_2, self.packindexfile_v2), (self.packfile_v2_3_ascii, self.packindexfile_v2_3_ascii)):
        packfile, version, size = packinfo
        indexfile, version, size = indexinfo
        entity = PackEntity(packfile)
        assert entity.pack().path() == packfile
        assert entity.index().path() == indexfile
        pack_objs.extend(entity.stream_iter())
        count = 0
        for info, stream in zip(entity.info_iter(), entity.stream_iter()):
            count += 1
            assert info.binsha == stream.binsha
            assert len(info.binsha) == 20
            assert info.type_id == stream.type_id
            assert info.size == stream.size
            assert not info.type_id in delta_types
            assert len(entity.collect_streams(info.binsha))
            oinfo = entity.info(info.binsha)
            assert isinstance(oinfo, OInfo)
            assert oinfo.binsha is not None
            ostream = entity.stream(info.binsha)
            assert isinstance(ostream, OStream)
            assert ostream.binsha is not None
            try:
                assert entity.is_valid_stream(info.binsha, use_crc=True)
            except UnsupportedOperation:
                pass
            assert entity.is_valid_stream(info.binsha, use_crc=False)
        assert count == size
    pack_path1 = tempfile.mktemp('', 'pack1', rw_dir)
    pack_path2 = tempfile.mktemp('', 'pack2', rw_dir)
    index_path = tempfile.mktemp('', 'index', rw_dir)
    iteration = 0

    def rewind_streams():
        for obj in pack_objs:
            obj.stream.seek(0)
    for ppath, ipath, num_obj in zip((pack_path1, pack_path2), (index_path, None), (len(pack_objs), None)):
        iwrite = None
        if ipath:
            ifile = open(ipath, 'wb')
            iwrite = ifile.write
        if iteration > 0:
            rewind_streams()
        iteration += 1
        with open(ppath, 'wb') as pfile:
            pack_sha, index_sha = PackEntity.write_pack(pack_objs, pfile.write, iwrite, object_count=num_obj)
        assert os.path.getsize(ppath) > 100
        pf = PackFile(ppath)
        assert pf.size() == len(pack_objs)
        assert pf.version() == PackFile.pack_version_default
        assert pf.checksum() == pack_sha
        pf.close()
        if ipath is not None:
            ifile.close()
            assert os.path.getsize(ipath) > 100
            idx = PackIndexFile(ipath)
            assert idx.version() == PackIndexFile.index_version_default
            assert idx.packfile_checksum() == pack_sha
            assert idx.indexfile_checksum() == index_sha
            assert idx.size() == len(pack_objs)
            idx.close()
    rewind_streams()
    entity = PackEntity.create(pack_objs, rw_dir)
    count = 0
    for info in entity.info_iter():
        count += 1
        for use_crc in range(2):
            assert entity.is_valid_stream(info.binsha, use_crc)
    assert count == len(pack_objs)
    entity.close()