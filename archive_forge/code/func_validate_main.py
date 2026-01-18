import argparse
import logging
import sys
from rdkit import Chem
from rdkit.Chem.MolStandardize import Standardizer, Validator
def validate_main(args):
    mol = _read_mol(args)
    v = Validator()
    logs = v.validate(mol)
    for log in logs:
        args.outfile.write(log)
        args.outfile.write('\n')