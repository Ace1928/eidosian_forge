import base64
import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from ... import branch, config, controldir, errors, repository, tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...branch import Branch
from ...revision import NULL_REVISION, Revision
from ...tests import test_server
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from ...transport.remote import (RemoteSSHTransport, RemoteTCPTransport,
from .. import (RemoteBzrProber, bzrdir, groupcompress_repo, inventory,
from ..bzrdir import BzrDir, BzrDirFormat
from ..chk_serializer import chk_bencode_serializer
from ..remote import (RemoteBranch, RemoteBranchFormat, RemoteBzrDir,
from ..smart import medium, request
from ..smart.client import _SmartClient
from ..smart.repository import (SmartServerRepositoryGetParentMap,
def test_stream_with_inventory_deltas(self):
    """'inventory-deltas' substreams cannot be sent to the
        Repository.insert_stream verb, because not all servers that implement
        that verb will accept them.  So when one is encountered the RemoteSink
        immediately stops using that verb and falls back to VFS insert_stream.
        """
    transport_path = 'quack'
    repo, client = self.setup_fake_client_and_repository(transport_path)
    client.add_expected_call(b'Repository.insert_stream_1.19', (b'quack/', b''), b'unknown', (b'Repository.insert_stream_1.19',))
    client.add_expected_call(b'Repository.insert_stream', (b'quack/', b''), b'success', (b'ok',))
    client.add_expected_call(b'Repository.insert_stream', (b'quack/', b''), b'success', (b'ok',))

    class FakeRealSink:

        def __init__(self):
            self.records = []

        def insert_stream(self, stream, src_format, resume_tokens):
            for substream_kind, substream in stream:
                self.records.append((substream_kind, [record.key for record in substream]))
            return ([b'fake tokens'], [b'fake missing keys'])
    fake_real_sink = FakeRealSink()

    class FakeRealRepository:

        def _get_sink(self):
            return fake_real_sink

        def is_in_write_group(self):
            return False

        def refresh_data(self):
            return True
    repo._real_repository = FakeRealRepository()
    sink = repo._get_sink()
    fmt = repository.format_registry.get_default()
    stream = self.make_stream_with_inv_deltas(fmt)
    resume_tokens, missing_keys = sink.insert_stream(stream, fmt, [])
    expected_records = [('inventory-deltas', [(b'rev2',), (b'rev3',)]), ('texts', [(b'some-rev', b'some-file')])]
    self.assertEqual(expected_records, fake_real_sink.records)
    self.assertEqual([b'fake tokens'], resume_tokens)
    self.assertEqual([b'fake missing keys'], missing_keys)
    self.assertFinished(client)