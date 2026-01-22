import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='import_from')
class _FutureImportRule(SyntaxRule):
    message = 'from __future__ imports must occur at the beginning of the file'

    def is_issue(self, node):
        if _is_future_import(node):
            if not _is_future_import_first(node):
                return True
            for from_name, future_name in node.get_paths():
                name = future_name.value
                allowed_futures = list(ALLOWED_FUTURES)
                if self._normalizer.version >= (3, 7):
                    allowed_futures.append('annotations')
                if name == 'braces':
                    self.add_issue(node, message='not a chance')
                elif name == 'barry_as_FLUFL':
                    m = "Seriously I'm not implementing this :) ~ Dave"
                    self.add_issue(node, message=m)
                elif name not in allowed_futures:
                    message = 'future feature %s is not defined' % name
                    self.add_issue(node, message=message)