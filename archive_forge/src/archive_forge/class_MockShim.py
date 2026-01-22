from typing import List
from thinc.shims.shim import Shim
from ..util import make_tempdir
class MockShim(Shim):

    def __init__(self, data: List[int]):
        super().__init__(None, config=None, optimizer=None)
        self.data = data

    def to_bytes(self):
        return bytes(self.data)

    def from_bytes(self, data: bytes) -> 'MockShim':
        return MockShim(data=list(data))