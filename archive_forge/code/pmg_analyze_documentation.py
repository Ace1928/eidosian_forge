from __future__ import annotations
import logging
import multiprocessing
import os
import re
from tabulate import tabulate
from pymatgen.apps.borg.hive import SimpleVaspToComputedEntryDrone, VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
from pymatgen.io.vasp import Outcar
Master function controlling which analysis to call.

    Args:
        args (dict): args from argparse.
    