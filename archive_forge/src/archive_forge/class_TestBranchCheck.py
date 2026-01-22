from io import BytesIO
from ... import errors, tests, ui
from . import TestCaseWithBranch
class TestBranchCheck(TestCaseWithBranch):

    def test_check_detects_invalid_revhistory(self):
        tree = self.make_branch_and_tree('test')
        r1 = tree.commit('one')
        r2 = tree.commit('two')
        r3 = tree.commit('three')
        r4 = tree.commit('four')
        tree.set_parent_ids([r1])
        tree.branch.set_last_revision_info(1, r1)
        r2b = tree.commit('two-b')
        tree.set_parent_ids([r4, r2b])
        tree.branch.set_last_revision_info(4, r4)
        r5 = tree.commit('five')
        if getattr(tree.branch, '_set_revision_history', None) is not None:
            tree.branch._set_revision_history([r1, r2b, r5])
        else:
            tree.branch.set_last_revision_info(3, r5)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        refs = self.make_refs(tree.branch)
        result = tree.branch.check(refs)
        ui.ui_factory = tests.TestUIFactory(stdout=BytesIO())
        result.report_results(True)
        self.assertContainsRe(b'revno does not match len', ui.ui_factory.stdout.getvalue())

    def test_check_branch_report_results(self):
        """Checking a branch produces results which can be printed"""
        branch = self.make_branch('.')
        branch.lock_read()
        self.addCleanup(branch.unlock)
        result = branch.check(self.make_refs(branch))
        result.report_results(verbose=True)
        result.report_results(verbose=False)

    def test__get_check_refs(self):
        tree = self.make_branch_and_tree('.')
        revid = tree.commit('foo')
        self.assertEqual({('revision-existence', revid), ('lefthand-distance', revid)}, set(tree.branch._get_check_refs()))

    def make_refs(self, branch):
        needed_refs = branch._get_check_refs()
        refs = {}
        distances = set()
        existences = set()
        for ref in needed_refs:
            kind, value = ref
            if kind == 'lefthand-distance':
                distances.add(value)
            elif kind == 'revision-existence':
                existences.add(value)
            else:
                raise AssertionError('unknown ref kind for ref %s' % ref)
        node_distances = branch.repository.get_graph().find_lefthand_distances(distances)
        for key, distance in node_distances.items():
            refs['lefthand-distance', key] = distance
            if key in existences and distance > 0:
                refs['revision-existence', key] = True
                existences.remove(key)
        parent_map = branch.repository.get_graph().get_parent_map(existences)
        for key in parent_map:
            refs['revision-existence', key] = True
            existences.remove(key)
        for key in existences:
            refs['revision-existence', key] = False
        return refs