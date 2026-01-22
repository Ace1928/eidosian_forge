import pytest
from ..tractogram import Tractogram
from ..tractogram_file import TractogramFile
class DummyTractogramFile(TractogramFile):

    @classmethod
    def is_correct_format(cls, fileobj):
        return False

    @classmethod
    def load(cls, fileobj, lazy_load=True):
        return None

    @classmethod
    def save(self, fileobj):
        pass