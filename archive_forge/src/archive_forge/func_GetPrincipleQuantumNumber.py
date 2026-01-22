import numpy
from rdkit import Chem
def GetPrincipleQuantumNumber(atNum):
    """ Get principal quantum number for atom number """
    if atNum <= 2:
        return 1
    if atNum <= 10:
        return 2
    if atNum <= 18:
        return 3
    if atNum <= 36:
        return 4
    if atNum <= 54:
        return 5
    if atNum <= 86:
        return 6
    return 7