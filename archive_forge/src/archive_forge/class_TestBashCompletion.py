import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
class TestBashCompletion(tests.TestCase, BashCompletionMixin):
    """Test bash completions that don't execute brz."""

    def test_simple_scipt(self):
        """Ensure that the test harness works as expected"""
        self.script = '\n_brz() {\n    COMPREPLY=()\n    # add all words in reverse order, with some markup around them\n    for ((i = ${#COMP_WORDS[@]}; i > 0; --i)); do\n        COMPREPLY+=( "-${COMP_WORDS[i-1]}+" )\n    done\n    # and append the current word\n    COMPREPLY+=( "+${COMP_WORDS[COMP_CWORD]}-" )\n}\n'
        self.complete(['foo', '"bar', "'baz"], cword=1)
        self.assertCompletionEquals("-'baz+", '-"bar+', '-foo+', '+"bar-')

    def test_cmd_ini(self):
        self.complete(['brz', 'ini'])
        self.assertCompletionContains('init', 'init-shared-repo', 'init-shared-repository')
        self.assertCompletionOmits('commit')

    def test_init_opts(self):
        self.complete(['brz', 'init', '-'])
        self.assertCompletionContains('-h', '--format=2a')

    def test_global_opts(self):
        self.complete(['brz', '-', 'init'], cword=1)
        self.assertCompletionContains('--no-plugins', '--builtin')

    def test_commit_dashm(self):
        self.complete(['brz', 'commit', '-m'])
        self.assertCompletionEquals('-m')

    def test_status_negated(self):
        self.complete(['brz', 'status', '--n'])
        self.assertCompletionContains('--no-versioned', '--no-verbose')

    def test_init_format_any(self):
        self.complete(['brz', 'init', '--format', '=', 'directory'], cword=3)
        self.assertCompletionContains('1.9', '2a')

    def test_init_format_2(self):
        self.complete(['brz', 'init', '--format', '=', '2', 'directory'], cword=4)
        self.assertCompletionContains('2a')
        self.assertCompletionOmits('1.9')