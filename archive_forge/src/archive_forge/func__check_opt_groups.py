import stevedore
from testtools import matchers
from glance_store import backend
from glance_store.tests import base
def _check_opt_groups(self, opt_list, expected_opt_groups):
    self.assertThat(opt_list, matchers.HasLength(len(expected_opt_groups)))
    groups = [g for g, _l in opt_list]
    self.assertThat(groups, matchers.HasLength(len(expected_opt_groups)))
    for idx, group in enumerate(groups):
        self.assertEqual(expected_opt_groups[idx], group)