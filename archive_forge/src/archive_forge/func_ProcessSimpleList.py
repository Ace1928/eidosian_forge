from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def ProcessSimpleList(self):
    """ Handles the list of simple descriptors

      This constructs the list of _nonZeroDescriptors_ and _requiredDescriptors_.

      There's some other magic going on that I can't decipher at the moment.

    """
    global countOptions
    self.nonZeroDescriptors = []
    lCopy = self.simpleList[:]
    tList = map(lambda x: x[0], countOptions)
    for entry in lCopy:
        if 'NONZERO' in entry[1]:
            if entry[0] not in tList:
                self.nonZeroDescriptors.append('%s != 0' % entry[0])
            if len(entry[1]) == 1:
                self.simpleList.remove(entry)
            else:
                self.simpleList[self.simpleList.index(entry)][1].remove('NONZERO')
    self.requiredDescriptors = map(lambda x: x[0], self.simpleList)
    for entry in tList:
        if entry in self.requiredDescriptors:
            self.requiredDescriptors.remove(entry)