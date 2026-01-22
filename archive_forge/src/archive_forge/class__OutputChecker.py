from __future__ import absolute_import, division, print_function
import doctest
import numpy as np
import pkg_resources
from . import audio, evaluation, features, io, ml, models, processors, utils
class _OutputChecker(_doctest_OutputChecker):
    """
    Output checker which enhances `doctest.OutputChecker` to compare doctests
    and computed output with additional flags.

    """

    def check_output(self, want, got, optionflags):
        """
        Return 'True' if the actual output from an example matches the
        expected.

        Parameters
        ----------
        want : str
            Expected output.
        got : str
            Actual output.
        optionflags : int
            Comparison flags.

        Returns
        -------
        bool
            'True' if the output maches the expectation.

        """
        import re
        if optionflags & _NORMALIZE_ARRAYS:
            got = re.sub('\\( ', '(', got)
            got = re.sub('\\[ ', '[', got)
            got = re.sub('0\\.0', '0.', got)
            got = re.sub('\\s*,', ',', got)
            want = re.sub('\\( ', '(', want)
            want = re.sub('\\[ ', '[', want)
            want = re.sub('0\\.0', '0.', want)
            want = re.sub('\\s*,', ',', want)
        super_check_output = _doctest_OutputChecker.check_output
        return super_check_output(self, want, got, optionflags)