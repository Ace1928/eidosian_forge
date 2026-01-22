import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def _test_pop_hz_both(self, fname, type, ext, enum_test=True, dememorization=10000, batches=20, iterations=5000):
    """Use Hardy-Weinberg test for heterozygote deficiency/excess (PRIVATE).

        Returns a population iterator containing a dictionary where
        dictionary[locus]=(P-val, SE, Fis-WC, Fis-RH, steps).

        Some loci have a None if the info is not available.
        SE might be none (for enumerations).
        """
    opts = self._get_opts(dememorization, batches, iterations, enum_test)
    self._run_genepop([ext], [1, type], fname, opts)

    def hw_func(self):
        return _hw_func(self.stream, False)
    return _FileIterator(hw_func, fname + ext)