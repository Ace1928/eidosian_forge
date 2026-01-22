import os
from breezy import tests
from breezy.tests import features
from breezy.transport import memory
class SelfTestPatch:

    def get_params_passed_to_core(self, cmdline):
        params = []

        def selftest(*args, **kwargs):
            """Capture the arguments selftest was run with."""
            params.append((args, kwargs))
            return True
        original_selftest = tests.selftest
        tests.selftest = selftest
        try:
            self.run_bzr(cmdline)
            return params[0]
        finally:
            tests.selftest = original_selftest