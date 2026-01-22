import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def _do_commit(self):
    builder = tests.GitBranchBuilder()
    builder.set_file(b'a', b'text for a\n', False)
    commit_handle = builder.commit(b'Joe Foo <joe@foo.com>', b'message')
    mapping = builder.finish()
    return mapping[commit_handle]