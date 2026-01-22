import argparse
import logging
import sys
from rdkit import Chem
from rdkit.Chem.MolStandardize import Standardizer, Validator
def _write_mol(mol, args):
    if args.outtype in {'smi', 'smiles'} or args.outfile.name.endswith('smi') or args.outfile.name.endswith('smiles'):
        args.outfile.write(Chem.MolToSmiles(mol))
        args.outfile.write('\n')
    elif args.outtype in {'mol', 'sdf'} or args.outfile.name.endswith('mol') or args.outfile.name.endswith('sdf'):
        args.outfile.write(Chem.MolToMolBlock(mol))
    else:
        args.outfile.write(Chem.MolToSmiles(mol))
        args.outfile.write('\n')