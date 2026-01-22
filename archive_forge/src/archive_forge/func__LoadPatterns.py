import os
from rdkit import Chem, RDConfig
def _LoadPatterns(fileName=None):
    if fileName is None:
        fileName = defaultPatternFileName
    try:
        with open(fileName, 'r') as inF:
            for line in inF.readlines():
                if len(line) and line[0] != '#':
                    splitL = line.split('\t')
                    if len(splitL) >= 3:
                        name = splitL[0]
                        descr = splitL[1]
                        sma = splitL[2]
                        descr = descr.replace('"', '')
                        patt = Chem.MolFromSmarts(sma)
                        if not patt or patt.GetNumAtoms() == 0:
                            raise ImportError('Smarts %s could not be parsed' % repr(sma))
                        fn = lambda mol, countUnique=True, pattern=patt: _CountMatches(mol, pattern, unique=countUnique)
                        fn.__doc__ = descr
                        name = name.replace('=', '_')
                        name = name.replace('-', '_')
                        fns.append((name, fn))
    except IOError:
        pass