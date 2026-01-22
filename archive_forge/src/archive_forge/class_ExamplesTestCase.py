import keyword
import os
import re
import subprocess
import sys
from taskflow import test
class ExamplesTestCase(test.TestCase, metaclass=ExampleAdderMeta):
    """Runs the examples, and checks the outputs against expected outputs."""

    def _check_example(self, name):
        output = run_example(name)
        eop = expected_output_path(name)
        if os.path.isfile(eop):
            with open(eop) as f:
                expected_output = f.read()
            output = UUID_RE.sub('<SOME UUID>', output)
            expected_output = UUID_RE.sub('<SOME UUID>', expected_output)
            self.assertEqual(expected_output, output)