import os
import pytest
@pytest.mark.xfail(os.getenv('PYTEST_XDIST_WORKER') is not None, reason='Execution in the same scope cannot be guaranteed')
class TestRunAllTests(ValidateAPI):
    """Class to test that each validator test gets run

    We check this in the module teardown function
    """
    run_tests = []

    def obj_params(self):
        yield (1, 2)

    def validate_first(self, obj, param):
        self.run_tests.append('first')

    def validate_second(self, obj, param):
        self.run_tests.append('second')

    @classmethod
    def teardown_class(cls):
        assert cls.run_tests == ['first', 'second']