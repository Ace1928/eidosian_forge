import pickle
from rdkit import Chem, DataStructs
class LayeredOptions:
    loadLayerFlags = 4294967295
    searchLayerFlags = 7
    minPath = 1
    maxPath = 6
    fpSize = 1024
    wordSize = 32
    nWords = fpSize // wordSize

    @staticmethod
    def GetFingerprint(mol, query=True):
        if query:
            flags = LayeredOptions.searchLayerFlags
        else:
            flags = LayeredOptions.loadLayerFlags
        return Chem.LayeredFingerprint(mol, layerFlags=flags, minPath=LayeredOptions.minPath, maxPath=LayeredOptions.maxPath, fpSize=LayeredOptions.fpSize)

    @staticmethod
    def GetWords(mol, query=True):
        txt = LayeredOptions.GetFingerprint(mol, query=query).ToBitString()
        return [int(txt[x:x + 32], 2) for x in range(0, len(txt), 32)]

    @staticmethod
    def GetQueryText(mol, query=True):
        words = LayeredOptions.GetWords(mol, query=query)
        colqs = []
        for idx, word in enumerate(words):
            if not word:
                continue
            colqs.append(f'{word}&Col_{idx + 1}={word}')
        return ' and '.join(colqs)