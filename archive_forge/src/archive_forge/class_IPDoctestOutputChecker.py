import doctest
import logging
import re
from testpath import modified_env
class IPDoctestOutputChecker(doctest.OutputChecker):
    """Second-chance checker with support for random tests.

    If the default comparison doesn't pass, this checker looks in the expected
    output string for flags that tell us to ignore the output.
    """
    random_re = re.compile('#\\s*random\\s+')

    def check_output(self, want, got, optionflags):
        """Check output, accepting special markers embedded in the output.

        If the output didn't pass the default validation but the special string
        '#random' is included, we accept it."""
        ret = doctest.OutputChecker.check_output(self, want, got, optionflags)
        if not ret and self.random_re.search(want):
            return True
        return ret