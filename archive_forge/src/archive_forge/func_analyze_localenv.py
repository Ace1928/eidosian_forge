from __future__ import annotations
from tabulate import tabulate
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def analyze_localenv(args):
    """Analyze local env of structures in files.

    Args:
        args (dict): Args for argparse.
    """
    bonds = {}
    for bond in args.localenv:
        tokens = bond.split('=')
        species = tokens[0].split('-')
        bonds[species[0], species[1]] = float(tokens[1])
    for filename in args.filenames:
        print(f'Analyzing {filename}...')
        data = []
        struct = Structure.from_file(filename)
        for idx, site in enumerate(struct):
            for species, dist in bonds.items():
                if species[0] in [sp.symbol for sp in site.species]:
                    dists = [nn.nn_distance for nn in struct.get_neighbors(site, dist) if species[1] in [sp.symbol for sp in nn.species]]
                    dists = ', '.join((f'{d:.3f}' for d in sorted(dists)))
                    data.append([idx, species[0], species[1], dists])
        print(tabulate(data, headers=['#', 'Center', 'Ligand', 'Dists']))