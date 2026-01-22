from __future__ import unicode_literals
import logging
import re
from cmakelang.parse import util as parse_util
from cmakelang.parse.funs import standard_funs
from cmakelang import markup
from cmakelang.config_util import (
class FormattingConfig(ConfigObject):
    """Options affecting formatting."""
    _field_registry = []
    disable = FieldDescriptor(False, 'Disable formatting entirely, making cmake-format a no-op')
    line_width = FieldDescriptor(80, 'How wide to allow formatted cmake files')
    tab_size = FieldDescriptor(2, 'How many spaces to tab for indent')
    use_tabchars = FieldDescriptor(False, 'If true, lines are indented using tab characters (utf-8 0x09) instead of <tab_size> space characters (utf-8 0x20). In cases where the layout would require a fractional tab character, the behavior of the  fractional indentation is governed by <fractional_tab_policy>')
    fractional_tab_policy = FieldDescriptor('use-space', "If <use_tabchars> is True, then the value of this variable indicates how fractional indentions are handled during whitespace replacement. If set to 'use-space', fractional indentation is left as spaces (utf-8 0x20). If set to `round-up` fractional indentation is replaced with a single tab character (utf-8 0x09) effectively shifting the column to the next tabstop", ['use-space', 'round-up'])
    max_subgroups_hwrap = FieldDescriptor(2, 'If an argument group contains more than this many sub-groups (parg or kwarg groups) then force it to a vertical layout. ')
    max_pargs_hwrap = FieldDescriptor(6, 'If a positional argument group contains more than this many arguments, then force it to a vertical layout. ')
    max_rows_cmdline = FieldDescriptor(2, 'If a cmdline positional group consumes more than this many lines without nesting, then invalidate the layout (and nest)')
    separate_ctrl_name_with_space = FieldDescriptor(False, 'If true, separate flow control names from their parentheses with a space')
    separate_fn_name_with_space = FieldDescriptor(False, 'If true, separate function names from parentheses with a space')
    dangle_parens = FieldDescriptor(False, 'If a statement is wrapped to more than one line, than dangle the closing parenthesis on its own line.')
    dangle_align = FieldDescriptor('prefix', "If the trailing parenthesis must be 'dangled' on its on line, then align it to this reference: `prefix`: the start of the statement,  `prefix-indent`: the start of the statement, plus one indentation  level, `child`: align to the column of the arguments", ['prefix', 'prefix-indent', 'child', 'off'])
    min_prefix_chars = FieldDescriptor(4, 'If the statement spelling length (including space and parenthesis) is smaller than this amount, then force reject nested layouts.')
    max_prefix_chars = FieldDescriptor(10, 'If the statement spelling length (including space and parenthesis) is larger than the tab width by more than this amount, then force reject un-nested layouts.')
    max_lines_hwrap = FieldDescriptor(2, 'If a candidate layout is wrapped horizontally but it exceeds this many lines, then reject the layout.')
    line_ending = FieldDescriptor('unix', 'What style line endings to use in the output.', ['windows', 'unix', 'auto'])
    command_case = FieldDescriptor('canonical', "Format command names consistently as 'lower' or 'upper' case", ['lower', 'upper', 'canonical', 'unchanged'])
    keyword_case = FieldDescriptor('unchanged', "Format keywords consistently as 'lower' or 'upper' case", ['lower', 'upper', 'unchanged'])
    always_wrap = FieldDescriptor([], 'A list of command names which should always be wrapped')
    enable_sort = FieldDescriptor(True, 'If true, the argument lists which are known to be sortable will be sorted lexicographicall')
    autosort = FieldDescriptor(False, 'If true, the parsers may infer whether or not an argument list is sortable (without annotation).')
    require_valid_layout = FieldDescriptor(False, 'By default, if cmake-format cannot successfully fit everything into the desired linewidth it will apply the last, most agressive attempt that it made. If this flag is True, however, cmake-format will print error, exit with non-zero status code, and write-out nothing')
    layout_passes = FieldDescriptor({}, 'A dictionary mapping layout nodes to a list of wrap decisions. See the documentation for more information.')

    def __init__(self, **kwargs):
        super(FormattingConfig, self).__init__(**kwargs)
        self.endl = None

    @property
    def linewidth(self):
        return self.line_width

    def set_line_ending(self, detected):
        self.endl = {'windows': '\r\n', 'unix': '\n'}[detected]

    def _update_derived(self):
        """Update derived values after a potential config change
    """
        self.endl = {'windows': '\r\n', 'unix': '\n', 'auto': '\n'}[self.line_ending]