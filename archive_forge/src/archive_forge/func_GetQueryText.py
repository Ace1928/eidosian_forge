import pickle
from rdkit import Chem, DataStructs
@staticmethod
def GetQueryText(mol, query=True):
    words = LayeredOptions.GetWords(mol, query=query)
    colqs = []
    for idx, word in enumerate(words):
        if not word:
            continue
        colqs.append(f'{word}&Col_{idx + 1}={word}')
    return ' and '.join(colqs)