import collections
from rdkit import Chem
from rdkit.Chem import AllChem
def EnumerateHeterocycles(inputmol, depth=None):
    """
    Enumerate possible relevant heterocycles on the given input
    molecule.

    >>> from rdkit.Chem.EnumerateHeterocycles import EnumerateHeterocycles
    >>> from rdkit import Chem
    >>> for smi in sorted(Chem.MolToSmiles(m) for m in EnumerateHeterocycles(Chem.MolFromSmiles('c1ccccc1'))):
    ...     print(smi)
    c1ccccc1
    c1ccncc1
    c1ccnnc1
    c1cnccn1
    c1cncnc1
    c1cnncn1
    c1cnnnc1
    c1ncncn1

    The algorithm works by mutating only one atom at a time. The depth
    parameter can be used to control the level of this recursion. For
    example, only enumerating aromatic rings that are one atom different:

    >>> for smi in sorted(Chem.MolToSmiles(m) for m in EnumerateHeterocycles(Chem.MolFromSmiles('n1ccccc1'), depth=1)):
    ...     print(smi)
    c1ccccc1
    c1ccnnc1
    c1cnccn1
    c1cncnc1
    """
    starting_points = [(inputmol, 0)]
    seen = set()
    while starting_points:
        curmol, curdepth = starting_points.pop(0)
        if depth is not None and curdepth >= depth:
            return
        for rxn in GetHeterocycleReactions():
            for newmol in rxn.RunReactants((curmol,)):
                newmol = newmol[0]
                Chem.SanitizeMol(newmol)
                newmol_smiles = Chem.MolToSmiles(newmol, isomericSmiles=True)
                if newmol_smiles in seen:
                    continue
                starting_points.append((newmol, curdepth + 1))
                seen.add(newmol_smiles)
                yield newmol