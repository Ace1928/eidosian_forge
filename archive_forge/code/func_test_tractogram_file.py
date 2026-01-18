import pytest
from ..tractogram import Tractogram
from ..tractogram_file import TractogramFile
def test_tractogram_file():
    with pytest.raises(NotImplementedError):
        TractogramFile.is_correct_format('')
    with pytest.raises(NotImplementedError):
        TractogramFile.load('')

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
    with pytest.raises(NotImplementedError):
        super(DummyTractogramFile, DummyTractogramFile(Tractogram)).save('')