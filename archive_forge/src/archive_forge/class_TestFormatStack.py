from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
class TestFormatStack(common.HeatTestCase):

    def setUp(self):
        super(TestFormatStack, self).setUp()
        self.request = mock.Mock()

    def test_doesnt_include_stack_action(self):
        stack = {'stack_action': 'CREATE'}
        result = stacks_view.format_stack(self.request, stack)
        self.assertEqual({}, result)

    def test_merges_stack_action_and_status(self):
        stack = {'stack_action': 'CREATE', 'stack_status': 'COMPLETE'}
        result = stacks_view.format_stack(self.request, stack)
        self.assertIn('stack_status', result)
        self.assertEqual('CREATE_COMPLETE', result['stack_status'])

    def test_include_stack_status_with_no_action(self):
        stack = {'stack_status': 'COMPLETE'}
        result = stacks_view.format_stack(self.request, stack)
        self.assertIn('stack_status', result)
        self.assertEqual('COMPLETE', result['stack_status'])

    @mock.patch.object(stacks_view, 'util')
    def test_replace_stack_identity_with_id_and_links(self, mock_util):
        mock_util.make_link.return_value = 'blah'
        stack = {'stack_identity': {'stack_id': 'foo'}}
        result = stacks_view.format_stack(self.request, stack)
        self.assertIn('id', result)
        self.assertNotIn('stack_identity', result)
        self.assertEqual('foo', result['id'])
        self.assertIn('links', result)
        self.assertEqual(['blah'], result['links'])

    @mock.patch.object(stacks_view, 'util', new=mock.Mock())
    def test_doesnt_add_project_by_default(self):
        stack = {'stack_identity': {'stack_id': 'foo', 'tenant': 'bar'}}
        result = stacks_view.format_stack(self.request, stack, None)
        self.assertNotIn('project', result)

    @mock.patch.object(stacks_view, 'util', new=mock.Mock())
    def test_doesnt_add_project_if_not_include_project(self):
        stack = {'stack_identity': {'stack_id': 'foo', 'tenant': 'bar'}}
        result = stacks_view.format_stack(self.request, stack, None, include_project=False)
        self.assertNotIn('project', result)

    @mock.patch.object(stacks_view, 'util', new=mock.Mock())
    def test_adds_project_if_include_project(self):
        stack = {'stack_identity': {'stack_id': 'foo', 'tenant': 'bar'}}
        result = stacks_view.format_stack(self.request, stack, None, include_project=True)
        self.assertIn('project', result)
        self.assertEqual('bar', result['project'])

    def test_includes_all_other_keys(self):
        stack = {'foo': 'bar'}
        result = stacks_view.format_stack(self.request, stack)
        self.assertIn('foo', result)
        self.assertEqual('bar', result['foo'])

    def test_filter_out_all_but_given_keys(self):
        stack = {'foo1': 'bar1', 'foo2': 'bar2', 'foo3': 'bar3'}
        result = stacks_view.format_stack(self.request, stack, ['foo2'])
        self.assertIn('foo2', result)
        self.assertNotIn('foo1', result)
        self.assertNotIn('foo3', result)