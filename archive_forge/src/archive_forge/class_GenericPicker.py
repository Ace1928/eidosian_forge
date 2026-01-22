import bisect
from rdkit import DataStructs
from rdkit.DataStructs.TopNContainer import TopNContainer
class GenericPicker(object):
    _picks = None

    def MakePicks(self, force=False):
        raise NotImplementedError('GenericPicker is a virtual base class')

    def __len__(self):
        if self._picks is None:
            self.MakePicks()
        return len(self._picks)

    def __getitem__(self, which):
        if self._picks is None:
            self.MakePicks()
        return self._picks[which]