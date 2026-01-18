import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_wrong_order(self):
    self.assertRaises(errors.CommandError, log._generate_all_revisions, self.branch, '5', '2.1.3', 'reverse', delayed_graph_generation=True)