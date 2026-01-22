import pickle
from rdkit import DataStructs
from rdkit.VLib.Node import VLibNode
class RandomAccessDbFpSupplier(DbFpSupplier):
    """ DbFp supplier supporting random access:

  >>> import os.path
  >>> from rdkit import RDConfig
  >>> from rdkit.Dbase.DbConnection import DbConnect
  >>> fName = RDConfig.RDTestDatabase
  >>> conn = DbConnect(fName,'simple_combined')
  >>> suppl = RandomAccessDbFpSupplier(conn.GetData())
  >>> len(suppl)
  12

  we can pull individual fingerprints:

  >>> fp = suppl[5]
  >>> fp.GetNumBits()
  128
  >>> fp.GetNumOnBits()
  54

  a standard loop over the fingerprints:

  >>> fps = []
  >>> for fp in suppl:
  ...   fps.append(fp)
  >>> len(fps)
  12

  or we can use an indexed loop:

  >>> fps = [None] * len(suppl)
  >>> for i in range(len(suppl)):
  ...   fps[i] = suppl[i]
  >>> len(fps)
  12

  """

    def __init__(self, *args, **kwargs):
        DbFpSupplier.__init__(self, *args, **kwargs)
        self.reset()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        newD = self._data[idx]
        return self._BuildFp(newD)

    def reset(self):
        self._pos = -1

    def NextItem(self):
        self._pos += 1
        res = None
        if self._pos < len(self):
            res = self[self._pos]
        return res