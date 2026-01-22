from .... import config, msgeditor
from ....tests import TestCaseWithTransport
from ... import commitfromnews
class TestCommitTemplate(TestCaseWithTransport):

    def capture_template(self, commit, message):
        self.commits.append(commit)
        self.messages.append(message)
        if message is None:
            message = 'let this commit succeed I command thee.'
        return message

    def enable_commitfromnews(self):
        stack = config.GlobalStack()
        stack.set('commit.template_from_files', ['NEWS'])

    def setup_capture(self):
        commitfromnews.register()
        msgeditor.hooks.install_named_hook('commit_message_template', self.capture_template, 'commitfromnews test template')
        self.messages = []
        self.commits = []

    def test_initial(self):
        self.setup_capture()
        self.enable_commitfromnews()
        builder = self.make_branch_builder('test')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', None, 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'a\nb\nc\nd\ne\n'))], message_callback=msgeditor.generate_commit_message_template, revision_id=b'BASE-id')
        builder.finish_series()
        self.assertEqual([None], self.messages)

    def test_added_NEWS(self):
        self.setup_capture()
        self.enable_commitfromnews()
        builder = self.make_branch_builder('test')
        builder.start_series()
        content = INITIAL_NEWS_CONTENT
        builder.build_snapshot(None, [('add', ('', None, 'directory', None)), ('add', ('NEWS', b'foo-id', 'file', content))], message_callback=msgeditor.generate_commit_message_template, revision_id=b'BASE-id')
        builder.finish_series()
        self.assertEqual([content.decode('utf-8')], self.messages)

    def test_changed_NEWS(self):
        self.setup_capture()
        self.enable_commitfromnews()
        builder = self.make_branch_builder('test')
        builder.start_series()
        orig_content = INITIAL_NEWS_CONTENT
        mod_content = b'----------------------------\ncommitfromnews release notes\n----------------------------\n\nNEXT (In development)\n---------------------\n\nIMPROVEMENTS\n~~~~~~~~~~~~\n\n* Added a new change to the system.\n\n* Created plugin, basic functionality of looking for NEWS and including the\n  NEWS diff.\n'
        change_content = '* Added a new change to the system.\n\n'
        builder.build_snapshot(None, [('add', ('', None, 'directory', None)), ('add', ('NEWS', b'foo-id', 'file', orig_content))], revision_id=b'BASE-id')
        builder.build_snapshot(None, [('modify', ('NEWS', mod_content))], message_callback=msgeditor.generate_commit_message_template)
        builder.finish_series()
        self.assertEqual([change_content], self.messages)

    def test_fix_bug(self):
        self.setup_capture()
        self.enable_commitfromnews()
        builder = self.make_branch_builder('test')
        builder.start_series()
        orig_content = INITIAL_NEWS_CONTENT
        mod_content = b'----------------------------\ncommitfromnews release notes\n----------------------------\n\nNEXT (In development)\n---------------------\n\nIMPROVEMENTS\n~~~~~~~~~~~~\n\n* Created plugin, basic functionality of looking for NEWS and including the\n  NEWS diff.\n\n* Fixed a horrible bug. (lp:523423)\n\n'
        change_content = '\n* Fixed a horrible bug. (lp:523423)\n\n'
        builder.build_snapshot(None, [('add', ('', None, 'directory', None)), ('add', ('NEWS', b'foo-id', 'file', orig_content))], revision_id=b'BASE-id')
        builder.build_snapshot(None, [('modify', ('NEWS', mod_content))], message_callback=msgeditor.generate_commit_message_template)
        builder.finish_series()
        self.assertEqual([change_content], self.messages)
        self.assertEqual(1, len(self.commits))
        self.assertEqual('https://launchpad.net/bugs/523423 fixed', self.commits[0].revprops['bugs'])

    def _todo_test_passes_messages_through(self):
        pass