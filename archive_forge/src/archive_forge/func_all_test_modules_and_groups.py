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
def all_test_modules_and_groups():
    groups = set()
    for testpath in all_test_modules():
        group = testpath.parent
        if group not in groups:
            yield group
            groups.add(group)
        yield testpath