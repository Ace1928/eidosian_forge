import os
import shutil
import tempfile
import unittest
import gevent
from gevent import monkey
from dulwich import client, index, objects, repo, server  # noqa: E402
from dulwich.contrib import swift  # noqa: E402
def test_push_branch(self):

    def determine_wants(*args, **kwargs):
        return {'refs/heads/mybranch': local_repo.refs['refs/heads/mybranch']}
    local_repo = repo.Repo.init(self.temp_d, mkdir=True)
    local_repo.do_commit('Test commit', 'fbo@localhost', ref='refs/heads/mybranch')
    sha = local_repo.refs.read_loose_ref('refs/heads/mybranch')
    swift.SwiftRepo.init_bare(self.scon, self.conf)
    tcp_client = client.TCPGitClient(self.server_address, port=self.port)
    tcp_client.send_pack('/fakerepo', determine_wants, local_repo.generate_pack_data)
    swift_repo = swift.SwiftRepo(self.fakerepo, self.conf)
    remote_sha = swift_repo.refs.read_loose_ref('refs/heads/mybranch')
    self.assertEqual(sha, remote_sha)