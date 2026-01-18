import pytest
import numpy as np
from ase import io
from ase.optimize import BFGS
from ase.build import bulk
def vasp_vol_relax():
    Al = bulk('Al', 'fcc', a=4.5, cubic=True)
    calc = factory.calc(xc='LDA', isif=7, nsw=5, ibrion=1, ediffg=-0.001, lwave=False, lcharg=False)
    Al.calc = calc
    Al.get_potential_energy()
    CONTCAR_Al = io.read('CONTCAR', format='vasp')
    print('Stress after relaxation:\n', calc.read_stress())
    print('Al cell post relaxation from calc:\n', calc.get_atoms().get_cell())
    print('Al cell post relaxation from atoms:\n', Al.get_cell())
    print('Al cell post relaxation from CONTCAR:\n', CONTCAR_Al.get_cell())
    assert (calc.get_atoms().get_cell() == CONTCAR_Al.get_cell()).all()
    assert (Al.get_cell() == CONTCAR_Al.get_cell()).all()
    return Al