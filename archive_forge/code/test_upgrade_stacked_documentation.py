from .. import check, controldir, errors, tests
from ..upgrade import upgrade
from .scenarios import load_tests_apply_scenarios
Correct checks when stacked-on repository is upgraded.

        We initially stack on a repo with the same rich root support,
        we then upgrade it and should fail, we then upgrade the overlaid
        repository.
        