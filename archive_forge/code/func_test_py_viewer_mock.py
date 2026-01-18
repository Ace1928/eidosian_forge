import sys
import pytest
from ase.io import read
from ase.visualize import view
from ase.visualize.external import PyViewer, CLIViewer
from ase.build import bulk
def test_py_viewer_mock(atoms, monkeypatch):

    def mock_view(self, atoms, repeat=None):
        print(f'viewing {atoms} with mock "{self.name}"')
        return (atoms, self.name)
    monkeypatch.setattr(PyViewer, 'sage', mock_view, raising=False)
    atoms1, name1 = view(atoms, viewer='sage')
    assert name1 == 'sage'
    assert atoms1 == atoms
    atoms2, name2 = view(atoms, viewer='sage', repeat=(2, 2, 2), block=True)
    assert name2 == 'sage'
    assert len(atoms2) == 8 * len(atoms)