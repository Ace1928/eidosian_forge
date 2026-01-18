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
@pytest.mark.parametrize('atoms', different_dimensionalities())
def test_quickinfo(gui, atoms):
    gui.new_atoms(atoms)
    refstring = _('Single image loaded.')
    infostring = info(gui)
    assert refstring in infostring
    dia = gui.quick_info_window()
    txt = dia.things[0].text
    assert refstring in txt