import argparse
import traceback
from math import pi
from time import time
import numpy as np
import ase.db
import ase.optimize
from ase.calculators.emt import EMT
from ase.io import Trajectory
def test_optimizer(systems, optimizer, calculator, prefix='', db=None, eggbox=0.0):
    """Test optimizer on systems."""
    for name, atoms in systems:
        if db is not None:
            optname = optimizer.__name__
            id = db.reserve(optimizer=optname, name=name)
            if id is None:
                continue
        atoms = atoms.copy()
        tag = '{}{}-{}'.format(prefix, optname, name)
        atoms.calc = calculator(txt=tag + '.txt')
        error, nsteps, texcl, tincl = run_test(atoms, optimizer, tag, eggbox=eggbox)
        if db is not None:
            db.write(atoms, id=id, optimizer=optname, name=name, error=error, n=nsteps, t=texcl, T=tincl, eggbox=eggbox)