import os
import stat
import sys
from io import BytesIO
from .. import errors, osutils, pyutils, tests
from .. import transport as _mod_transport
from .. import urlutils
from ..errors import ConnectionError, PathError, TransportNotPossible
from ..osutils import getcwd
from ..transport import (ConnectedTransport, FileExists, NoSuchFile, Transport,
from ..transport.memory import MemoryTransport
from ..transport.remote import RemoteTransport
from . import TestNotApplicable, TestSkipped, multiply_tests, test_server
from .test_transport import TestTransportImplementation
def test_readv_with_adjust_for_latency(self):
    transport = self.get_transport()
    content = osutils.rand_bytes(200 * 1024)
    content_size = len(content)
    if transport.is_readonly():
        self.build_tree_contents([('a', content)])
    else:
        transport.put_bytes('a', content)

    def check_result_data(result_vector):
        for item in result_vector:
            data_len = len(item[1])
            self.assertEqual(content[item[0]:item[0] + data_len], item[1])
    result = list(transport.readv('a', ((0, 30),), adjust_for_latency=True, upper_limit=content_size))
    self.assertEqual(1, len(result))
    self.assertEqual(0, result[0][0])
    self.assertTrue(len(result[0][1]) >= 30)
    check_result_data(result)
    result = list(transport.readv('a', ((204700, 100),), adjust_for_latency=True, upper_limit=content_size))
    self.assertEqual(1, len(result))
    data_len = len(result[0][1])
    self.assertEqual(204800 - data_len, result[0][0])
    self.assertTrue(data_len >= 100)
    check_result_data(result)
    result = list(transport.readv('a', ((204700, 100), (0, 50)), adjust_for_latency=True, upper_limit=content_size))
    self.assertEqual(2, len(result))
    data_len = len(result[0][1])
    self.assertEqual(0, result[0][0])
    self.assertTrue(data_len >= 30)
    data_len = len(result[1][1])
    self.assertEqual(204800 - data_len, result[1][0])
    self.assertTrue(data_len >= 100)
    check_result_data(result)
    for request_vector in [((400, 50), (800, 234)), ((800, 234), (400, 50))]:
        result = list(transport.readv('a', request_vector, adjust_for_latency=True, upper_limit=content_size))
        self.assertEqual(1, len(result))
        data_len = len(result[0][1])
        self.assertTrue(data_len >= 634)
        self.assertTrue(result[0][0] <= 400)
        self.assertTrue(result[0][0] + data_len >= 1034)
        check_result_data(result)