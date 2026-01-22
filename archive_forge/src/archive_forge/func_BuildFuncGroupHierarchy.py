import os
import re
import weakref
from rdkit import Chem, RDConfig
def BuildFuncGroupHierarchy(fileNm=None, data=None, force=False):
    global groupDefns, hierarchy, lastData, lastFilename
    if not force and hierarchy and (not data or data == lastData) and (not fileNm or fileNm == lastFilename):
        return hierarchy[:]
    lastData = data
    splitter = re.compile('\t+')
    if not fileNm and (not data):
        fileNm = os.path.join(RDConfig.RDDataDir, 'Functional_Group_Hierarchy.txt')
    if fileNm:
        with open(fileNm, 'r') as inF:
            data = inF.readlines()
        lastFilename = fileNm
    elif data:
        data = data.splitlines()
    else:
        raise ValueError('need data or filename')
    groupDefns = {}
    res = []
    for lineNo, line in enumerate(data, 1):
        line = line.strip()
        line = line.split('//')[0]
        if not line:
            continue
        splitL = splitter.split(line)
        if len(splitL) < 3:
            raise FuncGroupFileParseError('Input line %d (%s) is not long enough.' % (lineNo, repr(line)))
        label = splitL[0].strip()
        if label in groupDefns:
            raise FuncGroupFileParseError('Duplicate label on line %d.' % lineNo)
        labelHierarchy = label.split('.')
        if len(labelHierarchy) > 1:
            for i in range(len(labelHierarchy) - 1):
                tmp = '.'.join(labelHierarchy[:i + 1])
                if tmp not in groupDefns:
                    raise FuncGroupFileParseError('Hierarchy member %s (line %d) not found.' % (tmp, lineNo))
            parent = groupDefns['.'.join(labelHierarchy[:-1])]
        else:
            parent = None
        smarts = splitL[1]
        patt = Chem.MolFromSmarts(smarts)
        if not patt:
            raise FuncGroupFileParseError('Smarts "%s" (line %d) could not be parsed.' % (smarts, lineNo))
        name = splitL[2].strip()
        rxnSmarts = ''
        if len(splitL) > 3:
            rxnSmarts = splitL[3]
        node = FGHierarchyNode(name, patt, smarts=smarts, label=label, parent=parent, rxnSmarts=rxnSmarts)
        if parent:
            parent.children.append(node)
        else:
            res.append(node)
        groupDefns[label] = node
    hierarchy = res[:]
    return res