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
def test_open_and_save(gui, testdir):
    mol = molecule('H2O')
    for i in range(3):
        mol.write('h2o.json')
    gui.open(filename='h2o.json')
    save_dialog(gui, 'h2o.cif@-1')