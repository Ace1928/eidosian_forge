import os
import shutil
import tempfile
import unittest
import gevent
from gevent import monkey
from dulwich import client, index, objects, repo, server  # noqa: E402
from dulwich.contrib import swift  # noqa: E402
def test_push_data_branch(self):

    def determine_wants(*args, **kwargs):
        return {'refs/heads/master': local_repo.refs['HEAD']}
    local_repo = repo.Repo.init(self.temp_d, mkdir=True)
    os.mkdir(os.path.join(self.temp_d, 'dir'))
    files = ('testfile', 'testfile2', 'dir/testfile3')
    i = 0
    for f in files:
        open(os.path.join(self.temp_d, f), 'w').write('DATA %s' % i)
        i += 1
    local_repo.stage(files)
    local_repo.do_commit('Test commit', 'fbo@localhost', ref='refs/heads/master')
    swift.SwiftRepo.init_bare(self.scon, self.conf)
    tcp_client = client.TCPGitClient(self.server_address, port=self.port)
    tcp_client.send_pack(self.fakerepo, determine_wants, local_repo.generate_pack_data)
    swift_repo = swift.SwiftRepo('fakerepo', self.conf)
    commit_sha = swift_repo.refs.read_loose_ref('refs/heads/master')
    otype, data = swift_repo.object_store.get_raw(commit_sha)
    commit = objects.ShaFile.from_raw_string(otype, data)
    otype, data = swift_repo.object_store.get_raw(commit._tree)
    tree = objects.ShaFile.from_raw_string(otype, data)
    objs = tree.items()
    objs_ = []
    for tree_entry in objs:
        objs_.append(swift_repo.object_store.get_raw(tree_entry.sha))
    self.assertEqual(objs_[1][1], 'DATA 0')
    self.assertEqual(objs_[2][1], 'DATA 1')
    self.assertEqual(objs_[0][0], 2)