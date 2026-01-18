import os
from io import BytesIO
from ... import (branch, builtins, check, controldir, errors, push, revision,
from ...bzr import branch as bzrbranch
from ...bzr.smart import client
from .. import per_branch, test_server
def test_push_new_tag_to_bound_branch(self):
    master = self.make_branch('master')
    bound = self.make_branch('bound')
    try:
        bound.bind(master)
    except branch.BindingUnsupported:
        raise tests.TestNotApplicable('Format does not support bound branches')
    other = bound.controldir.sprout('other').open_branch()
    try:
        other.tags.set_tag('new-tag', b'some-rev')
    except errors.TagsNotSupported:
        raise tests.TestNotApplicable('Format does not support tags')
    other.push(bound)
    self.assertEqual({'new-tag': b'some-rev'}, bound.tags.get_tag_dict())
    self.assertEqual({'new-tag': b'some-rev'}, master.tags.get_tag_dict())