import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from breezy import branch as _mod_branch
from breezy import controldir, errors, gpg, tests, transport, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import inventory_delta, versionedfile
from breezy.bzr.smart import branch as smart_branch
from breezy.bzr.smart import bzrdir as smart_dir
from breezy.bzr.smart import packrepository as smart_packrepo
from breezy.bzr.smart import repository as smart_repo
from breezy.bzr.smart import request as smart_req
from breezy.bzr.smart import server, vfs
from breezy.bzr.testament import Testament
from breezy.tests import test_server
from breezy.transport import chroot, memory
def _get_serialized_inventory_delta(self, repository, base_revid, revid):
    base_inv = repository.revision_tree(base_revid).root_inventory
    inv = repository.revision_tree(revid).root_inventory
    inv_delta = inv._make_delta(base_inv)
    serializer = inventory_delta.InventoryDeltaSerializer(True, True)
    return b''.join(serializer.delta_to_lines(base_revid, revid, inv_delta))