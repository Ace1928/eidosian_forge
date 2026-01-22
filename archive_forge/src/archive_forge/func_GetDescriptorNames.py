from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def GetDescriptorNames(self):
    """ returns a list of the names of the descriptors this calculator generates

    """
    if self.descriptorNames is not None:
        return self.descriptorNames
    else:
        res = []
        for descName, targets in self.simpleList:
            for target in targets:
                if hasattr(self, target):
                    res.append('%s_%s' % (target, descName))
                else:
                    print('Method %s does not exist' % target)
        for entry in self.compoundList:
            res.append(entry[0])
        self.descriptorNames = res[:]
        return tuple(res)