from __future__ import unicode_literals
import logging
import re
from cmakelang.parse import util as parse_util
from cmakelang.parse.funs import standard_funs
from cmakelang import markup
from cmakelang.config_util import (
class MarkupConfig(ConfigObject):
    """Options affecting comment reflow and formatting."""
    _field_registry = []
    bullet_char = FieldDescriptor('*', 'What character to use for bulleted lists')
    enum_char = FieldDescriptor('.', 'What character to use as punctuation after numerals in an enumerated list')
    first_comment_is_literal = FieldDescriptor(False, "If comment markup is enabled, don't reflow the first comment block in each listfile. Use this to preserve formatting of your copyright/license statements. ")
    literal_comment_pattern = FieldDescriptor(None, "If comment markup is enabled, don't reflow any comment block which matches this (regex) pattern. Default is `None` (disabled).")
    fence_pattern = FieldDescriptor(markup.FENCE_PATTERN, "Regular expression to match preformat fences in comments default= ``r'{}'``".format(markup.FENCE_PATTERN))
    ruler_pattern = FieldDescriptor(markup.RULER_PATTERN, "Regular expression to match rulers in comments default= ``r'{}'``".format(markup.RULER_PATTERN))
    explicit_trailing_pattern = FieldDescriptor('#<', "If a comment line matches starts with this pattern then it is explicitly a trailing comment for the preceeding argument. Default is '#<'")
    hashruler_min_length = FieldDescriptor(10, "If a comment line starts with at least this many consecutive hash characters, then don't lstrip() them off. This allows for lazy hash rulers where the first hash char is not separated by space")
    canonicalize_hashrulers = FieldDescriptor(True, 'If true, then insert a space between the first hash char and remaining hash chars in a hash ruler, and normalize its length to fill the column')
    enable_markup = FieldDescriptor(True, 'enable comment markup parsing and reflow')