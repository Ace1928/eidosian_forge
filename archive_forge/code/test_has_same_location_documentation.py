from breezy import controldir, transport
from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
Different format repositories are comparable and not the same.

        Comparing different format repository objects should give a negative
        result, rather than trigger an exception (which could happen with a
        naive __eq__ implementation, e.g. due to missing attributes).
        