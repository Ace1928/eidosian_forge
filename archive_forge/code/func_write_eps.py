import time
from ase.utils import writer
from ase.io.utils import PlottingVariables, make_patch_list
@writer
def write_eps(fd, atoms, **parameters):
    EPS(atoms, **parameters).write(fd)