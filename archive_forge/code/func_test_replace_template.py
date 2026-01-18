import itertools
from heat.scaling import template
from heat.tests import common
def test_replace_template(self):
    """Test case for replacing template.

        If num_replace is the number of old resources, then all of the
        resources will be replaced.
        """
    old_resources = [('old-id-0', {'type': 'Foo'}), ('old-id-1', {'type': 'Foo'})]
    templates = template.member_definitions(old_resources, {'type': 'Bar'}, 1, 2, self.next_id)
    expected = [('old-id-1', {'type': 'Bar'})]
    self.assertEqual(expected, list(templates))