import itertools
from heat.scaling import template
from heat.tests import common
def test_replace_units_some_already_up_to_date(self):
    """Test case for up-to-date resources in template.

        If some of the old resources already have the new resource definition,
        then they won't be considered for replacement, and the next resource
        that is out-of-date will be replaced.
        """
    old_resources = [('old-id-0', {'type': 'Bar'}), ('old-id-1', {'type': 'Foo'})]
    new_spec = {'type': 'Bar'}
    templates = template.member_definitions(old_resources, new_spec, 2, 1, self.next_id)
    second_batch_expected = [('old-id-0', {'type': 'Bar'}), ('old-id-1', {'type': 'Bar'})]
    self.assertEqual(second_batch_expected, list(templates))