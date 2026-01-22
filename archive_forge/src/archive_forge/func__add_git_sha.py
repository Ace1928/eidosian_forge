import os
import threading
from dulwich.objects import ShaFile, hex_to_sha, sha_to_hex
from .. import bedding
from .. import errors as bzr_errors
from .. import osutils, registry, trace
from ..bzr import btree_index as _mod_btree_index
from ..bzr import index as _mod_index
from ..bzr import versionedfile
from ..transport import FileExists, NoSuchFile, get_transport_from_path
def _add_git_sha(self, hexsha, type, type_data):
    if hexsha is not None:
        self._name.update(hexsha)
        if type == b'commit':
            td = (type_data[0], type_data[1])
            try:
                td += (type_data[2]['testament3-sha1'],)
            except KeyError:
                pass
        else:
            td = type_data
        self._add_node((b'git', hexsha, b'X'), b' '.join((type,) + td))
    else:
        self._name.update(type + b' '.join(type_data))