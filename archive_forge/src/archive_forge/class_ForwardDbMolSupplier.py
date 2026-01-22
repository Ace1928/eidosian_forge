import sys
from rdkit import Chem
from rdkit.Chem.Suppliers.MolSupplier import MolSupplier
class ForwardDbMolSupplier(DbMolSupplier):
    """ DbMol supplier supporting only forward iteration


    new molecules come back with all additional fields from the
    database set in a "_fieldsFromDb" data member

  """

    def __init__(self, dbResults, **kwargs):
        """

      DbResults should be an iterator for Dbase.DbResultSet.DbResultBase

    """
        DbMolSupplier.__init__(self, dbResults, **kwargs)
        self.Reset()

    def Reset(self):
        self._dataIter = iter(self._data)

    def NextMol(self):
        """

      NOTE: this has side effects

    """
        try:
            d = self._dataIter.next()
        except StopIteration:
            d = None
        if d is not None:
            newM = self._BuildMol(d)
        else:
            newM = None
        return newM