import numpy as np
from ase import Atoms
from ase.calculators.acn import (ACN, m_me, r_cn, r_mec,
from ase.calculators.qmmm import SimpleQMMM, EIQMMM, LJInteractionsGeneral
from ase.md.verlet import VelocityVerlet
from ase.constraints import FixLinearTriatomic
import ase.units as units
Test RATTLE and QM/MM for rigid linear acetonitrile.