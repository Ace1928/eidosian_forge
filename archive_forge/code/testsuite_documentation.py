import os
import sys
from subprocess import Popen
import importlib
from pathlib import Path
import warnings
import argparse
from multiprocessing import cpu_count
from ase.calculators.calculator import names as calc_names
from ase.cli.main import CLIError
Run ASE's test-suite.

    Requires the pytest package.  pytest-xdist is recommended
    in addition as the tests will then run in parallel.
    