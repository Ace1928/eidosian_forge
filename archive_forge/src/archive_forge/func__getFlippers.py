import random
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
def _getFlippers(mol, options):
    Chem.FindPotentialStereoBonds(mol)
    flippers = []
    if not options.onlyStereoGroups:
        for atom in mol.GetAtoms():
            if atom.HasProp('_ChiralityPossible'):
                if not options.onlyUnassigned or atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                    flippers.append(_AtomFlipper(atom))
        for bond in mol.GetBonds():
            bstereo = bond.GetStereo()
            if bstereo != Chem.BondStereo.STEREONONE:
                if not options.onlyUnassigned or bstereo == Chem.BondStereo.STEREOANY:
                    flippers.append(_BondFlipper(bond))
    if options.onlyUnassigned:
        for group in mol.GetStereoGroups():
            if group.GetGroupType() != Chem.StereoGroupType.STEREO_ABSOLUTE:
                flippers.append(_StereoGroupFlipper(group))
    return flippers