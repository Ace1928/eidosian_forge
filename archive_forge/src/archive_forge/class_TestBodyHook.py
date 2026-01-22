import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
class TestBodyHook(tests.TestCaseWithTransport):

    def compose_with_hooks(self, test_hooks, supports_body=True):
        client = HookMailClient({})
        client.supports_body = supports_body
        for test_hook in test_hooks:
            merge_directive.MergeDirective.hooks.install_named_hook('merge_request_body', test_hook, 'test')
        tree = self.make_branch_and_tree('foo')
        tree.commit('foo')
        directive = merge_directive.MergeDirective2(tree.branch.last_revision(), b'sha', 0, 0, b'sha', source_branch=tree.branch.base, base_revision_id=tree.branch.last_revision(), message='This code rox')
        directive.compose_merge_request(client, 'jrandom@example.com', None, tree.branch)
        return (client, directive)

    def test_no_supports_body(self):
        test_hook = TestHook('foo')
        old_warn = trace.warning
        warnings = []

        def warn(*args):
            warnings.append(args)
        trace.warning = warn
        try:
            client, directive = self.compose_with_hooks([test_hook], supports_body=False)
        finally:
            trace.warning = old_warn
        self.assertEqual(0, len(test_hook.calls))
        self.assertEqual(('Cannot run merge_request_body hooks because mail client %s does not support message bodies.', 'HookMailClient'), warnings[0])

    def test_body_hook(self):
        test_hook = TestHook('foo')
        client, directive = self.compose_with_hooks([test_hook])
        self.assertEqual(1, len(test_hook.calls))
        self.assertEqual('foo', client.body)
        params = test_hook.calls[0]
        self.assertIsInstance(params, merge_directive.MergeRequestBodyParams)
        self.assertIs(None, params.body)
        self.assertIs(None, params.orig_body)
        self.assertEqual('jrandom@example.com', params.to)
        self.assertEqual('[MERGE] This code rox', params.subject)
        self.assertEqual(directive, params.directive)
        self.assertEqual('foo-1', params.basename)

    def test_body_hook_chaining(self):
        test_hook1 = TestHook('foo')
        test_hook2 = TestHook('bar')
        client = self.compose_with_hooks([test_hook1, test_hook2])[0]
        self.assertEqual(None, test_hook1.calls[0].body)
        self.assertEqual(None, test_hook1.calls[0].orig_body)
        self.assertEqual('foo', test_hook2.calls[0].body)
        self.assertEqual(None, test_hook2.calls[0].orig_body)
        self.assertEqual('bar', client.body)