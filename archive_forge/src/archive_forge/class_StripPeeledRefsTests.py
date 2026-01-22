import os
import sys
import tempfile
from io import BytesIO
from typing import ClassVar, Dict
from dulwich import errors
from dulwich.tests import SkipTest, TestCase
from ..file import GitFile
from ..objects import ZERO_SHA
from ..refs import (
from ..repo import Repo
from .utils import open_repo, tear_down_repo
class StripPeeledRefsTests(TestCase):
    all_refs: ClassVar[Dict[bytes, bytes]] = {b'refs/heads/master': b'8843d7f92416211de9ebb963ff4ce28125932878', b'refs/heads/testing': b'186a005b134d8639a58b6731c7c1ea821a6eedba', b'refs/tags/1.0.0': b'a93db4b0360cc635a2b93675010bac8d101f73f0', b'refs/tags/1.0.0^{}': b'a93db4b0360cc635a2b93675010bac8d101f73f0', b'refs/tags/2.0.0': b'0749936d0956c661ac8f8d3483774509c165f89e', b'refs/tags/2.0.0^{}': b'0749936d0956c661ac8f8d3483774509c165f89e'}
    non_peeled_refs: ClassVar[Dict[bytes, bytes]] = {b'refs/heads/master': b'8843d7f92416211de9ebb963ff4ce28125932878', b'refs/heads/testing': b'186a005b134d8639a58b6731c7c1ea821a6eedba', b'refs/tags/1.0.0': b'a93db4b0360cc635a2b93675010bac8d101f73f0', b'refs/tags/2.0.0': b'0749936d0956c661ac8f8d3483774509c165f89e'}

    def test_strip_peeled_refs(self):
        self.assertEqual(strip_peeled_refs(self.all_refs), self.non_peeled_refs)