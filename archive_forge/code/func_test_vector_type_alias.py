import numpy as np
from numba.core import config
from numba.cuda.testing import CUDATestCase
from numba import cuda
def test_vector_type_alias(self):
    """Tests that `cuda.<vector_type.alias>` are importable and
        that is the same as `cuda.<vector_type.name>`.

        `test_fancy_creation_readout` only test vector types imported
        with its name. This test makes sure that construction with
        objects imported with alias should work the same.
        """
    for vty in vector_types.values():
        for alias in vty.user_facing_object.aliases:
            with self.subTest(vty=vty.name, alias=alias):
                self.assertEqual(id(getattr(cuda, vty.name)), id(getattr(cuda, alias)))