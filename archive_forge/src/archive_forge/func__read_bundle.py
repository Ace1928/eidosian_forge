from typing import Dict, List, Optional, Sequence, Tuple, Union
from .pack import PackData, write_pack_data
def _read_bundle(f, version):
    capabilities = {}
    prerequisites = []
    references = {}
    line = f.readline()
    if version >= 3:
        while line.startswith(b'@'):
            line = line[1:].rstrip(b'\n')
            try:
                key, value = line.split(b'=', 1)
            except ValueError:
                key = line
                value = None
            else:
                value = value.decode('utf-8')
            capabilities[key.decode('utf-8')] = value
            line = f.readline()
    while line.startswith(b'-'):
        obj_id, comment = line[1:].rstrip(b'\n').split(b' ', 1)
        prerequisites.append((obj_id, comment.decode('utf-8')))
        line = f.readline()
    while line != b'\n':
        obj_id, ref = line.rstrip(b'\n').split(b' ', 1)
        references[ref] = obj_id
        line = f.readline()
    pack_data = PackData.from_file(f)
    ret = Bundle()
    ret.references = references
    ret.capabilities = capabilities
    ret.prerequisites = prerequisites
    ret.pack_data = pack_data
    ret.version = version
    return ret