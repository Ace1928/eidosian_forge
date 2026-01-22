from rdkit import RDLogger as logging
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem.ChemUtils.AlignDepict import AlignDepict
def Explode(template, sidechains, outF, autoNames=True, do3D=False, useTethers=False):
    chainIndices = []
    core = Chem.DeleteSubstructs(template, Chem.MolFromSmiles('[*]'))
    try:
        templateName = template.GetProp('_Name')
    except KeyError:
        templateName = 'template'
    for mol in _exploder(template, 0, sidechains, core, chainIndices, autoNames=autoNames, templateName=templateName, do3D=do3D, useTethers=useTethers):
        outF.write(Chem.MolToMolBlock(mol))
        for pN in mol.GetPropNames():
            print('>  <%s>\n%s\n' % (pN, mol.GetProp(pN)), file=outF)
        print('$$$$', file=outF)