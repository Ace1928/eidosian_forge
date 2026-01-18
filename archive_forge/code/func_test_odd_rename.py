from ...controldir import format_registry
from ...repository import InterRepository
from ...tests import TestCaseWithTransport
from ..interrepo import InterToGitRepository
from ..mapping import BzrGitMappingExperimental, BzrGitMappingv1
def test_odd_rename(self):
    branch = self.bzr_repo.controldir.create_branch()
    tree = branch.controldir.create_workingtree()
    self.build_tree(['bzr/bar/', 'bzr/bar/foobar'])
    tree.add(['bar', 'bar/foobar'])
    tree.commit('initial')
    self.build_tree(['bzr/baz/'])
    tree.add(['baz'])
    tree.rename_one('bar', 'baz/IrcDotNet')
    last_revid = tree.commit('rename')

    def decide(x):
        return {b'refs/heads/master': (None, last_revid)}
    interrepo = self._get_interrepo()
    revidmap, old_refs, new_refs = interrepo.fetch_refs(decide, lossy=True)
    gitid = revidmap[last_revid][0]
    store = self.git_repo._git.object_store
    commit = store[gitid]
    tree = store[commit.tree]
    tree.check()
    self.assertIn(b'baz', tree, repr(tree.items()))
    self.assertIn(tree[b'baz'][1], store)
    baz = store[tree[b'baz'][1]]
    baz.check()
    ircdotnet = store[baz[b'IrcDotNet'][1]]
    ircdotnet.check()
    foobar = store[ircdotnet[b'foobar'][1]]
    foobar.check()