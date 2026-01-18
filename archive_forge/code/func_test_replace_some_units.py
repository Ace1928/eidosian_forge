import itertools
from heat.scaling import template
from heat.tests import common
def test_replace_some_units(self):
    """Test case for making only the number of replacements specified.

        If the resource definition changes, only the number of replacements
        specified will be made; beyond that, the original templates are used.
        """
    old_resources = [('old-id-0', {'type': 'Foo'}), ('old-id-1', {'type': 'Foo'})]
    new_spec = {'type': 'Bar'}
    templates = template.member_definitions(old_resources, new_spec, 2, 1, self.next_id)
    expected = [('old-id-0', {'type': 'Bar'}), ('old-id-1', {'type': 'Foo'})]
    self.assertEqual(expected, list(templates))