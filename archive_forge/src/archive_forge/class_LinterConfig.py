from __future__ import unicode_literals
import logging
import re
from cmakelang.parse import util as parse_util
from cmakelang.parse.funs import standard_funs
from cmakelang import markup
from cmakelang.config_util import (
class LinterConfig(ConfigObject):
    """Options affecting the linter"""
    _field_registry = []
    disabled_codes = FieldDescriptor([], 'a list of lint codes to disable')
    function_pattern = FieldDescriptor('[0-9a-z_]+', 'regular expression pattern describing valid function names')
    macro_pattern = FieldDescriptor('[0-9A-Z_]+', 'regular expression pattern describing valid macro names')
    global_var_pattern = FieldDescriptor('[A-Z][0-9A-Z_]+', 'regular expression pattern describing valid names for variables with global (cache) scope')
    internal_var_pattern = FieldDescriptor('_[A-Z][0-9A-Z_]+', 'regular expression pattern describing valid names for variables with global scope (but internal semantic)')
    local_var_pattern = FieldDescriptor('[a-z][a-z0-9_]+', 'regular expression pattern describing valid names for variables with local scope')
    private_var_pattern = FieldDescriptor('_[0-9a-z_]+', 'regular expression pattern describing valid names for privatedirectory variables')
    public_var_pattern = FieldDescriptor('[A-Z][0-9A-Z_]+', 'regular expression pattern describing valid names for public directory variables')
    argument_var_pattern = FieldDescriptor('[a-z][a-z0-9_]+', 'regular expression pattern describing valid names for function/macro arguments and loop variables.')
    keyword_pattern = FieldDescriptor('[A-Z][0-9A-Z_]+', 'regular expression pattern describing valid names for keywords used in functions or macros')
    max_conditionals_custom_parser = FieldDescriptor(2, 'In the heuristic for C0201, how many conditionals to match within a loop in before considering the loop a parser.')
    min_statement_spacing = FieldDescriptor(1, 'Require at least this many newlines between statements')
    max_statement_spacing = FieldDescriptor(2, 'Require no more than this many newlines between statements')
    max_returns = FieldDescriptor(6)
    max_branches = FieldDescriptor(12)
    max_arguments = FieldDescriptor(5)
    max_localvars = FieldDescriptor(15)
    max_statements = FieldDescriptor(50)