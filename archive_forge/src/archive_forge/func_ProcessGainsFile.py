import sys
from rdkit import Chem
from rdkit.Chem.rdfragcatalog import *
def ProcessGainsFile(fileName, nToDo=-1, delim=',', haveDescriptions=1):
    inFile = open(fileName, 'r')
    nRead = 0
    res = []
    for line in inFile.xreadlines():
        nRead += 1
        splitL = [x.strip() for x in line.split(delim)]
        if nRead != 1 and len(splitL):
            bit = BitGainsInfo()
            bit.id = int(splitL[0])
            col = 1
            if haveDescriptions:
                bit.description = splitL[col]
                col += 1
            bit.gain = float(splitL[col])
            col += 1
            nPerClass = []
            for entry in splitL[col:]:
                nPerClass.append(int(entry))
            bit.nPerClass = nPerClass
            res.append(bit)
            if len(res) == nToDo:
                break
    return res