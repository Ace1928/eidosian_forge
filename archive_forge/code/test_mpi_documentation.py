import sys
from subprocess import run
Try to import all ASE modules and check that ase.parallel.world has not
    been used.  We want to delay use of world until after MPI4PY has been
    imported.

    We run the test in a subprocess so that we have a clean Python interpreter.