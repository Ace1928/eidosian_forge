import pytest
from ..tractogram import Tractogram
from ..tractogram_file import TractogramFile
def test_subclassing_tractogram_file():

    class DummyTractogramFile(TractogramFile):

        @classmethod
        def is_correct_format(cls, fileobj):
            return False

        @classmethod
        def load(cls, fileobj, lazy_load=True):
            return None

        @classmethod
        def create_empty_header(cls):
            return None
    with pytest.raises(TypeError):
        DummyTractogramFile(Tractogram())

    class DummyTractogramFile(TractogramFile):

        @classmethod
        def is_correct_format(cls, fileobj):
            return False

        def save(self, fileobj):
            pass

        @classmethod
        def create_empty_header(cls):
            return None
    with pytest.raises(TypeError):
        DummyTractogramFile(Tractogram())

    class DummyTractogramFile(TractogramFile):

        @classmethod
        def is_correct_format(cls, fileobj):
            return False

        @classmethod
        def load(cls, fileobj, lazy_load=True):
            return None

        def save(self, fileobj):
            pass
    dtf = DummyTractogramFile(Tractogram())
    assert dtf.header == {}