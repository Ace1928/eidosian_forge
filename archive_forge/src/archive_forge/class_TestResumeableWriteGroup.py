import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestResumeableWriteGroup(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def make_write_locked_repo(self, relpath='repo'):
        repo = self.make_repository(relpath)
        repo.lock_write()
        self.addCleanup(repo.unlock)
        return repo

    def reopen_repo(self, repo):
        same_repo = repo.controldir.open_repository()
        same_repo.lock_write()
        self.addCleanup(same_repo.unlock)
        return same_repo

    def require_suspendable_write_groups(self, reason):
        repo = self.make_repository('__suspend_test')
        repo.lock_write()
        self.addCleanup(repo.unlock)
        repo.start_write_group()
        try:
            wg_tokens = repo.suspend_write_group()
        except errors.UnsuspendableWriteGroup:
            repo.abort_write_group()
            raise tests.TestNotApplicable(reason)

    def test_suspend_write_group(self):
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        repo.texts.add_lines((b'file-id', b'revid'), (), [b'lines'])
        try:
            wg_tokens = repo.suspend_write_group()
        except errors.UnsuspendableWriteGroup:
            self.assertTrue(repo.is_in_write_group())
            repo.abort_write_group()
        else:
            self.assertFalse(repo.is_in_write_group())
            self.assertEqual(1, len(wg_tokens))
            self.assertIsInstance(wg_tokens[0], str)

    def test_resume_write_group_then_abort(self):
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        text_key = (b'file-id', b'revid')
        repo.texts.add_lines(text_key, (), [b'lines'])
        try:
            wg_tokens = repo.suspend_write_group()
        except errors.UnsuspendableWriteGroup:
            repo.abort_write_group()
            self.assertRaises(errors.UnsuspendableWriteGroup, repo.resume_write_group, [])
        else:
            same_repo = self.reopen_repo(repo)
            same_repo.resume_write_group(wg_tokens)
            self.assertEqual([text_key], list(same_repo.texts.keys()))
            self.assertTrue(same_repo.is_in_write_group())
            same_repo.abort_write_group()
            self.assertEqual([], list(repo.texts.keys()))

    def test_multiple_resume_write_group(self):
        self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        first_key = (b'file-id', b'revid')
        repo.texts.add_lines(first_key, (), [b'lines'])
        wg_tokens = repo.suspend_write_group()
        same_repo = self.reopen_repo(repo)
        same_repo.resume_write_group(wg_tokens)
        self.assertTrue(same_repo.is_in_write_group())
        second_key = (b'file-id', b'second-revid')
        same_repo.texts.add_lines(second_key, (first_key,), [b'more lines'])
        try:
            new_wg_tokens = same_repo.suspend_write_group()
        except:
            same_repo.abort_write_group(suppress_errors=True)
            raise
        self.assertEqual(2, len(new_wg_tokens))
        self.assertSubset(wg_tokens, new_wg_tokens)
        same_repo = self.reopen_repo(repo)
        same_repo.resume_write_group(new_wg_tokens)
        both_keys = {first_key, second_key}
        self.assertEqual(both_keys, same_repo.texts.keys())
        same_repo.abort_write_group()

    def test_no_op_suspend_resume(self):
        self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        text_key = (b'file-id', b'revid')
        repo.texts.add_lines(text_key, (), [b'lines'])
        wg_tokens = repo.suspend_write_group()
        same_repo = self.reopen_repo(repo)
        same_repo.resume_write_group(wg_tokens)
        new_wg_tokens = same_repo.suspend_write_group()
        self.assertEqual(wg_tokens, new_wg_tokens)
        same_repo = self.reopen_repo(repo)
        same_repo.resume_write_group(wg_tokens)
        self.assertEqual([text_key], list(same_repo.texts.keys()))
        same_repo.abort_write_group()

    def test_read_after_suspend_fails(self):
        self.require_suspendable_write_groups('Cannot test suspend on repo that does not support suspending')
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        text_key = (b'file-id', b'revid')
        repo.texts.add_lines(text_key, (), [b'lines'])
        wg_tokens = repo.suspend_write_group()
        self.assertEqual([], list(repo.texts.keys()))

    def test_read_after_second_suspend_fails(self):
        self.require_suspendable_write_groups('Cannot test suspend on repo that does not support suspending')
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        text_key = (b'file-id', b'revid')
        repo.texts.add_lines(text_key, (), [b'lines'])
        wg_tokens = repo.suspend_write_group()
        same_repo = self.reopen_repo(repo)
        same_repo.resume_write_group(wg_tokens)
        same_repo.suspend_write_group()
        self.assertEqual([], list(same_repo.texts.keys()))

    def test_read_after_resume_abort_fails(self):
        self.require_suspendable_write_groups('Cannot test suspend on repo that does not support suspending')
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        text_key = (b'file-id', b'revid')
        repo.texts.add_lines(text_key, (), [b'lines'])
        wg_tokens = repo.suspend_write_group()
        same_repo = self.reopen_repo(repo)
        same_repo.resume_write_group(wg_tokens)
        same_repo.abort_write_group()
        self.assertEqual([], list(same_repo.texts.keys()))

    def test_cannot_resume_aborted_write_group(self):
        self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        text_key = (b'file-id', b'revid')
        repo.texts.add_lines(text_key, (), [b'lines'])
        wg_tokens = repo.suspend_write_group()
        same_repo = self.reopen_repo(repo)
        same_repo.resume_write_group(wg_tokens)
        same_repo.abort_write_group()
        same_repo = self.reopen_repo(repo)
        self.assertRaises(errors.UnresumableWriteGroup, same_repo.resume_write_group, wg_tokens)

    def test_commit_resumed_write_group_no_new_data(self):
        self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        text_key = (b'file-id', b'revid')
        repo.texts.add_lines(text_key, (), [b'lines'])
        wg_tokens = repo.suspend_write_group()
        same_repo = self.reopen_repo(repo)
        same_repo.resume_write_group(wg_tokens)
        same_repo.commit_write_group()
        self.assertEqual([text_key], list(same_repo.texts.keys()))
        self.assertEqual(b'lines', next(same_repo.texts.get_record_stream([text_key], 'unordered', True)).get_bytes_as('fulltext'))
        self.assertRaises(errors.UnresumableWriteGroup, same_repo.resume_write_group, wg_tokens)

    def test_commit_resumed_write_group_plus_new_data(self):
        self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        first_key = (b'file-id', b'revid')
        repo.texts.add_lines(first_key, (), [b'lines'])
        wg_tokens = repo.suspend_write_group()
        same_repo = self.reopen_repo(repo)
        same_repo.resume_write_group(wg_tokens)
        second_key = (b'file-id', b'second-revid')
        same_repo.texts.add_lines(second_key, (first_key,), [b'more lines'])
        same_repo.commit_write_group()
        self.assertEqual({first_key, second_key}, set(same_repo.texts.keys()))
        self.assertEqual(b'lines', next(same_repo.texts.get_record_stream([first_key], 'unordered', True)).get_bytes_as('fulltext'))
        self.assertEqual(b'more lines', next(same_repo.texts.get_record_stream([second_key], 'unordered', True)).get_bytes_as('fulltext'))

    def make_source_with_delta_record(self):
        source_repo = self.make_write_locked_repo('source')
        source_repo.start_write_group()
        key_base = (b'file-id', b'base')
        key_delta = (b'file-id', b'delta')

        def text_stream():
            yield versionedfile.FulltextContentFactory(key_base, (), None, b'lines\n')
            yield versionedfile.FulltextContentFactory(key_delta, (key_base,), None, b'more\nlines\n')
        source_repo.texts.insert_record_stream(text_stream())
        source_repo.commit_write_group()
        return source_repo

    def test_commit_resumed_write_group_with_missing_parents(self):
        self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
        source_repo = self.make_source_with_delta_record()
        key_base = (b'file-id', b'base')
        key_delta = (b'file-id', b'delta')
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        stream = source_repo.texts.get_record_stream([key_delta], 'unordered', False)
        repo.texts.insert_record_stream(stream)
        try:
            repo.commit_write_group()
        except errors.BzrCheckError:
            pass
        else:
            same_repo = self.reopen_repo(repo)
            same_repo.lock_read()
            record = next(same_repo.texts.get_record_stream([key_delta], 'unordered', True))
            self.assertEqual(b'more\nlines\n', record.get_bytes_as('fulltext'))
            return
        wg_tokens = repo.suspend_write_group()
        same_repo = self.reopen_repo(repo)
        same_repo.resume_write_group(wg_tokens)
        self.assertRaises(errors.BzrCheckError, same_repo.commit_write_group)
        same_repo.abort_write_group()

    def test_commit_resumed_write_group_adding_missing_parents(self):
        self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
        source_repo = self.make_source_with_delta_record()
        key_base = (b'file-id', b'base')
        key_delta = (b'file-id', b'delta')
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        text_key = (b'file-id', b'revid')
        repo.texts.add_lines(text_key, (), [b'lines'])
        wg_tokens = repo.suspend_write_group()
        same_repo = self.reopen_repo(repo)
        same_repo.resume_write_group(wg_tokens)
        stream = source_repo.texts.get_record_stream([key_delta], 'unordered', False)
        same_repo.texts.insert_record_stream(stream)
        try:
            same_repo.commit_write_group()
        except errors.BzrCheckError:
            pass
        else:
            same_repo = self.reopen_repo(repo)
            same_repo.lock_read()
            record = next(same_repo.texts.get_record_stream([key_delta], 'unordered', True))
            self.assertEqual(b'more\nlines\n', record.get_bytes_as('fulltext'))
            return
        same_repo.abort_write_group()

    def test_add_missing_parent_after_resume(self):
        self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
        source_repo = self.make_source_with_delta_record()
        key_base = (b'file-id', b'base')
        key_delta = (b'file-id', b'delta')
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        stream = source_repo.texts.get_record_stream([key_delta], 'unordered', False)
        repo.texts.insert_record_stream(stream)
        wg_tokens = repo.suspend_write_group()
        same_repo = self.reopen_repo(repo)
        same_repo.resume_write_group(wg_tokens)
        stream = source_repo.texts.get_record_stream([key_base], 'unordered', False)
        same_repo.texts.insert_record_stream(stream)
        same_repo.commit_write_group()

    def test_suspend_empty_initial_write_group(self):
        """Suspending a write group with no writes returns an empty token
        list.
        """
        self.require_suspendable_write_groups('Cannot test suspend on repo that does not support suspending')
        repo = self.make_write_locked_repo()
        repo.start_write_group()
        wg_tokens = repo.suspend_write_group()
        self.assertEqual([], wg_tokens)

    def test_resume_empty_initial_write_group(self):
        """Resuming an empty token list is equivalent to start_write_group."""
        self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
        repo = self.make_write_locked_repo()
        repo.resume_write_group([])
        repo.abort_write_group()