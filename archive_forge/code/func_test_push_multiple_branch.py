import os
import shutil
import tempfile
import unittest
import gevent
from gevent import monkey
from dulwich import client, index, objects, repo, server  # noqa: E402
from dulwich.contrib import swift  # noqa: E402
def test_push_multiple_branch(self):

    def determine_wants(*args, **kwargs):
        return {'refs/heads/mybranch': local_repo.refs['refs/heads/mybranch'], 'refs/heads/master': local_repo.refs['refs/heads/master'], 'refs/heads/pullr-108': local_repo.refs['refs/heads/pullr-108']}
    local_repo = repo.Repo.init(self.temp_d, mkdir=True)
    local_shas = {}
    remote_shas = {}
    for branch in ('master', 'mybranch', 'pullr-108'):
        local_shas[branch] = local_repo.do_commit('Test commit %s' % branch, 'fbo@localhost', ref='refs/heads/%s' % branch)
    swift.SwiftRepo.init_bare(self.scon, self.conf)
    tcp_client = client.TCPGitClient(self.server_address, port=self.port)
    tcp_client.send_pack(self.fakerepo, determine_wants, local_repo.generate_pack_data)
    swift_repo = swift.SwiftRepo('fakerepo', self.conf)
    for branch in ('master', 'mybranch', 'pullr-108'):
        remote_shas[branch] = swift_repo.refs.read_loose_ref('refs/heads/%s' % branch)
    self.assertDictEqual(local_shas, remote_shas)