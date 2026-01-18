import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_revspec_tag_spaces(self):
    self.requireFeature(features.sed_feature)
    wt = self.make_branch_and_tree('.', format='dirstate-tags')
    wt.branch.tags.set_tag('tag with spaces', b'null:')
    self.complete(['brz', 'log', '-r', 'tag', ':', 't'])
    self.assertCompletionEquals('tag\\ with\\ spaces')
    self.complete(['brz', 'log', '-r', '"tag:t'])
    self.assertCompletionEquals('tag:tag with spaces')
    self.complete(['brz', 'log', '-r', "'tag:t"])
    self.assertCompletionEquals('tag:tag with spaces')