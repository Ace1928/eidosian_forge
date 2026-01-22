import unittest
from rdkit import Chem
from rdkit.Chem.MolStandardize.standardize import Standardizer
class FakeStandardizer(Standardizer):

    def normalize(self):

        def fake_normalize(y):
            props = y.GetPropsAsDict()
            for k, v in props:
                y.ClearProp(k)
            return y
        return fake_normalize