import os
from breezy import branch, errors
from breezy import merge as _mod_merge
from breezy import switch, tests, workingtree
def test_switch_configurable_file_merger(self):

    class DummyMerger(_mod_merge.ConfigurableFileMerger):
        name_prefix = 'file'
    _mod_merge.Merger.hooks.install_named_hook('merge_file_content', DummyMerger, 'test factory')
    foo = self.make_branch('foo')
    checkout = foo.create_checkout('checkout', lightweight=True)
    self.build_tree_contents([('checkout/file', b'a')])
    checkout.add('file')
    checkout.commit('a')
    bar = foo.controldir.sprout('bar').open_workingtree()
    self.build_tree_contents([('bar/file', b'b')])
    bar.commit('b')
    self.build_tree_contents([('checkout/file', b'c')])
    switch.switch(checkout.controldir, bar.branch)