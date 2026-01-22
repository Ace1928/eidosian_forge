from rdkit import DataStructs
from rdkit.DataStructs import TopNContainer
def GetSingleFingerprint(self, probe):
    """ returns a fingerprint for a single probe object

         This is potentially useful in initializing our internal
         probe object.

        """
    return self.fingerprinter(probe)