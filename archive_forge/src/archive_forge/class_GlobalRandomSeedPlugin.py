from os import environ
from random import Random
import pytest
class GlobalRandomSeedPlugin:

    @pytest.fixture(params=random_seeds)
    def global_random_seed(self, request):
        """Fixture to ask for a random yet controllable random seed.

            All tests that use this fixture accept the contract that they should
            deterministically pass for any seed value from 0 to 99 included.

            See the documentation for the SKLEARN_TESTS_GLOBAL_RANDOM_SEED
            variable for insrtuctions on how to use this fixture.

            https://scikit-learn.org/dev/computing/parallelism.html#sklearn-tests-global-random-seed
            """
        yield request.param