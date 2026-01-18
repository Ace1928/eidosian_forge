import gzip
import os
import re
import tempfile
from .... import tests
from ....tests import features
from ....tests.blackbox import ExternalBase
from ..cmds import _get_source_stream
from . import FastimportFeature
from :1
from :2
from :1
from :2
def test_tag_rewriting(self):
    tree = self.make_branch_and_tree('br')
    tree.commit('pointless')
    self.assertTrue(tree.branch.supports_tags())
    rev_id = tree.branch.dotted_revno_to_revision_id((1,))
    tree.branch.tags.set_tag('goodTag', rev_id)
    tree.branch.tags.set_tag('bad Tag', rev_id)
    data = self.run_bzr('fast-export --plain --no-rewrite-tag-names br')[0]
    self.assertNotEqual(-1, data.find('reset refs/tags/goodTag'))
    self.assertEqual(data.find('reset refs/tags/'), data.rfind('reset refs/tags/'))
    data = self.run_bzr('fast-export --plain --rewrite-tag-names br')[0]
    self.assertNotEqual(-1, data.find('reset refs/tags/goodTag'))
    self.assertNotEqual(-1, data.find('reset refs/tags/bad_Tag'))