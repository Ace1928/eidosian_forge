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
def test_repeated_substreams_same_kind_are_one_stream(self):
    stream = [('text', [versionedfile.FulltextContentFactory((b'k1',), None, None, b'foo')]), ('text', [versionedfile.FulltextContentFactory((b'k2',), None, None, b'bar')])]
    fmt = controldir.format_registry.get('pack-0.92')().repository_format
    bytes = smart_repo._stream_to_byte_stream(stream, fmt)
    streams = []
    fmt, stream = smart_repo._byte_stream_to_stream(bytes)
    for kind, substream in stream:
        streams.append((kind, list(substream)))
    self.assertLength(1, streams)
    self.assertLength(2, streams[0][1])