import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
class PEP8NormalizerConfig(ErrorFinderConfig):
    normalizer_class = PEP8Normalizer
    '\n    Normalizing to PEP8. Not really implemented, yet.\n    '

    def __init__(self, indentation=' ' * 4, hanging_indentation=None, max_characters=79, spaces_before_comment=2):
        self.indentation = indentation
        if hanging_indentation is None:
            hanging_indentation = indentation
        self.hanging_indentation = hanging_indentation
        self.closing_bracket_hanging_indentation = ''
        self.break_after_binary = False
        self.max_characters = max_characters
        self.spaces_before_comment = spaces_before_comment