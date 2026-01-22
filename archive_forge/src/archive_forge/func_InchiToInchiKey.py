import logging
from rdkit import RDLogger
from rdkit.Chem import rdinchi
def InchiToInchiKey(inchi):
    """Return the InChI key for the given InChI string. Return None on error"""
    ret = rdinchi.InchiToInchiKey(inchi)
    if ret:
        return ret
    else:
        return None