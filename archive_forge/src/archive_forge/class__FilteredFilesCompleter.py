from __future__ import absolute_import, division, print_function, unicode_literals
import os
import subprocess
from .compat import str, sys_encoding
class _FilteredFilesCompleter(object):

    def __init__(self, predicate):
        """
        Create the completer

        A predicate accepts as its only argument a candidate path and either
        accepts it or rejects it.
        """
        assert predicate, 'Expected a callable predicate'
        self.predicate = predicate

    def __call__(self, prefix, **kwargs):
        """
        Provide completions on prefix
        """
        target_dir = os.path.dirname(prefix)
        try:
            names = os.listdir(target_dir or '.')
        except:
            return
        incomplete_part = os.path.basename(prefix)
        for name in names:
            if not name.startswith(incomplete_part):
                continue
            candidate = os.path.join(target_dir, name)
            if not self.predicate(candidate):
                continue
            yield (candidate + '/' if os.path.isdir(candidate) else candidate)