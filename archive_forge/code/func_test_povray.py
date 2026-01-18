import os
from pathlib import Path
import pytest
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import molecule, bulk
import ase.gui.ui as ui
from ase.gui.i18n import _
from ase.gui.gui import GUI
from ase.gui.save import save_dialog
from ase.gui.quickinfo import info
def test_povray(gui, testdir):
    mol = molecule('H2O')
    gui.new_atoms(mol)
    n = gui.render_window()
    assert n.basename_widget.value == 'H2O'
    n.run_povray_widget.check.deselect()
    n.keep_files_widget.check.select()
    n.ok()
    ini = Path('./H2O.ini')
    pov = Path('./H2O.pov')
    assert ini.is_file()
    assert pov.is_file()
    with open(ini, 'r') as _:
        _ = _.read()
        assert 'H2O' in _
    with open(pov, 'r') as _:
        _ = _.read()
        assert 'atom' in _