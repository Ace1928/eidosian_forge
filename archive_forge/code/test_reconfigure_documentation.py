from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
Test a fairly realistic scenario for stacking:

         * make a branch with some history
         * branch it
         * make the second branch stacked on the first
         * commit in the second
         * then make the second unstacked, so it has to fill in history from
           the original fallback lying underneath its original content

        See discussion in <https://bugs.launchpad.net/bzr/+bug/391411>
        