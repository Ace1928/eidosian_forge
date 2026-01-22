import os
import dulwich
from dulwich.objects import Commit, Tag
from dulwich.repo import Repo as GitRepo
from ... import errors, revision, urlutils
from ...branch import Branch, InterBranch, UnstackableBranchFormat
from ...controldir import ControlDir
from ...repository import Repository
from .. import branch, tests
from ..dir import LocalGitControlDirFormat
from ..mapping import default_mapping
class BranchTests(tests.TestCaseInTempDir):

    def make_onerev_branch(self):
        os.mkdir('d')
        os.chdir('d')
        GitRepo.init('.')
        bb = tests.GitBranchBuilder()
        bb.set_file('foobar', b'foo\nbar\n', False)
        mark = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
        gitsha = bb.finish()[mark]
        os.chdir('..')
        return (os.path.abspath('d'), gitsha)

    def make_tworev_branch(self):
        os.mkdir('d')
        os.chdir('d')
        GitRepo.init('.')
        bb = tests.GitBranchBuilder()
        bb.set_file('foobar', b'foo\nbar\n', False)
        mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
        mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
        marks = bb.finish()
        os.chdir('..')
        return ('d', (marks[mark1], marks[mark2]))

    def clone_git_branch(self, from_url, to_url):
        from_dir = ControlDir.open(from_url)
        to_dir = from_dir.sprout(to_url)
        return to_dir.open_branch()

    def test_single_rev(self):
        path, gitsha = self.make_onerev_branch()
        oldrepo = Repository.open(path)
        revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha)
        self.assertEqual(gitsha, oldrepo._git.get_refs()[b'refs/heads/master'])
        newbranch = self.clone_git_branch(path, 'f')
        self.assertEqual([revid], newbranch.repository.all_revision_ids())

    def test_sprouted_tags(self):
        path, gitsha = self.make_onerev_branch()
        r = GitRepo(path)
        self.addCleanup(r.close)
        r.refs[b'refs/tags/lala'] = r.head()
        oldrepo = Repository.open(path)
        revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha)
        newbranch = self.clone_git_branch(path, 'f')
        self.assertEqual({'lala': revid}, newbranch.tags.get_tag_dict())
        self.assertEqual([revid], newbranch.repository.all_revision_ids())

    def test_sprouted_ghost_tags(self):
        path, gitsha = self.make_onerev_branch()
        r = GitRepo(path)
        self.addCleanup(r.close)
        r.refs[b'refs/tags/lala'] = b'aa' * 20
        oldrepo = Repository.open(path)
        revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha)
        warnings, newbranch = self.callCatchWarnings(self.clone_git_branch, path, 'f')
        self.assertEqual({}, newbranch.tags.get_tag_dict())
        self.assertIn(('ref refs/tags/lala points at non-present sha ' + 'aa' * 20,), [w.args for w in warnings])

    def test_interbranch_pull_submodule(self):
        path = 'd'
        os.mkdir(path)
        os.chdir(path)
        GitRepo.init('.')
        bb = tests.GitBranchBuilder()
        bb.set_file('foobar', b'foo\nbar\n', False)
        mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
        bb.set_submodule('core', b'102ee7206ebc4227bec8ac02450972e6738f4a33')
        bb.set_file('.gitmodules', b'[submodule "core"]\n  path = core\n  url = https://github.com/phhusson/QuasselC.git\n', False)
        mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
        marks = bb.finish()
        os.chdir('..')
        gitsha1 = marks[mark1]
        gitsha2 = marks[mark2]
        oldrepo = Repository.open(path)
        revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
        newbranch = self.make_branch('g')
        inter_branch = InterBranch.get(Branch.open(path), newbranch)
        inter_branch.pull()
        self.assertEqual(revid2, newbranch.last_revision())
        self.assertEqual(('https://github.com/phhusson/QuasselC.git', 'core'), newbranch.get_reference_info(newbranch.basis_tree().path2id('core')))

    def test_interbranch_pull(self):
        path, (gitsha1, gitsha2) = self.make_tworev_branch()
        oldrepo = Repository.open(path)
        revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
        newbranch = self.make_branch('g')
        inter_branch = InterBranch.get(Branch.open(path), newbranch)
        inter_branch.pull()
        self.assertEqual(revid2, newbranch.last_revision())

    def test_interbranch_pull_noop(self):
        path, (gitsha1, gitsha2) = self.make_tworev_branch()
        oldrepo = Repository.open(path)
        revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
        newbranch = self.make_branch('g')
        inter_branch = InterBranch.get(Branch.open(path), newbranch)
        inter_branch.pull()
        inter_branch.pull()
        self.assertEqual(revid2, newbranch.last_revision())

    def test_interbranch_pull_stop_revision(self):
        path, (gitsha1, gitsha2) = self.make_tworev_branch()
        oldrepo = Repository.open(path)
        revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
        newbranch = self.make_branch('g')
        inter_branch = InterBranch.get(Branch.open(path), newbranch)
        inter_branch.pull(stop_revision=revid1)
        self.assertEqual(revid1, newbranch.last_revision())

    def test_interbranch_pull_with_tags(self):
        path, (gitsha1, gitsha2) = self.make_tworev_branch()
        gitrepo = GitRepo(path)
        self.addCleanup(gitrepo.close)
        gitrepo.refs[b'refs/tags/sometag'] = gitsha2
        oldrepo = Repository.open(path)
        revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
        revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
        newbranch = self.make_branch('g')
        source_branch = Branch.open(path)
        source_branch.get_config().set_user_option('branch.fetch_tags', True)
        inter_branch = InterBranch.get(source_branch, newbranch)
        inter_branch.pull(stop_revision=revid1)
        self.assertEqual(revid1, newbranch.last_revision())
        self.assertTrue(newbranch.repository.has_revision(revid2))

    def test_bzr_branch_bound_to_git(self):
        path, (gitsha1, gitsha2) = self.make_tworev_branch()
        wt = Branch.open(path).create_checkout('co')
        self.build_tree_contents([('co/foobar', b'blah')])
        self.assertRaises(errors.NoRoundtrippingSupport, wt.commit, 'commit from bound branch.')
        revid = wt.commit('commit from bound branch.', lossy=True)
        self.assertEqual(revid, wt.branch.last_revision())
        self.assertEqual(revid, wt.branch.get_master_branch().last_revision())

    def test_interbranch_pull_older(self):
        path, (gitsha1, gitsha2) = self.make_tworev_branch()
        oldrepo = Repository.open(path)
        revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
        revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
        newbranch = self.make_branch('g')
        inter_branch = InterBranch.get(Branch.open(path), newbranch)
        inter_branch.pull(stop_revision=revid2)
        inter_branch.pull(stop_revision=revid1)
        self.assertEqual(revid2, newbranch.last_revision())