from __future__ import annotations
import multiprocessing as multiproc
import warnings
from string import ascii_uppercase
from time import time
from typing import TYPE_CHECKING
from pymatgen.command_line.mcsqs_caller import Sqs
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
def _get_best_sqs_from_list(self, structures: list[Atoms], output_list: list[dict]) -> None:
    """
        Find best SQS structure from list of SQS structures.

        Args:
            structures (list of ase Atoms) : list of SQS structures
            output_list (list of dicts) : shared list between
                multiprocessing processes to store best SQS objects.
        """
    best_sqs: dict[str, Any] = {'structure': None, 'objective_function': 1e+20}
    cluster_space = self._get_cluster_space()
    for structure in structures:
        objective = self.get_icet_sqs_obj(structure, cluster_space=cluster_space)
        if objective < best_sqs['objective_function']:
            best_sqs = {'structure': structure, 'objective_function': objective}
    output_list.append(best_sqs)