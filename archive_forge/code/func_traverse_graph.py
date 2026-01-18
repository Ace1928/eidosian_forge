import itertools
import collections
import numpy as np
from ase import Atoms
from ase.geometry.cell import complete_cell
from ase.geometry.dimensionality import analyze_dimensionality
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality.bond_generator import next_bond
from ase.geometry.dimensionality.interval_analysis import merge_intervals
def traverse_graph(atoms, kcutoff):
    if kcutoff is None:
        kcutoff = select_cutoff(atoms)
    rda = rank_determination.RDA(len(atoms))
    for k, i, j, offset in next_bond(atoms):
        if k > kcutoff:
            break
        rda.insert_bond(i, j, offset)
    rda.check()
    return (rda.graph.find_all(), rda.all_visited, rda.ranks)