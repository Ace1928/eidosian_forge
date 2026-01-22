from ..core.options import Options as BaseOptions
class BeautifierOptions(BaseOptions):

    def __init__(self, options=None):
        BaseOptions.__init__(self, options, 'js')
        self.css = None
        self.js = None
        self.html = None
        raw_brace_style = getattr(self.raw_options, 'brace_style', None)
        if raw_brace_style == 'expand-strict':
            setattr(self.raw_options, 'brace_style', 'expand')
        elif raw_brace_style == 'collapse-preserve-inline':
            setattr(self.raw_options, 'brace_style', 'collapse,preserve-inline')
        brace_style_split = self._get_selection_list('brace_style', ['collapse', 'expand', 'end-expand', 'none', 'preserve-inline'])
        self.brace_preserve_inline = False
        self.brace_style = 'collapse'
        for bs in brace_style_split:
            if bs == 'preserve-inline':
                self.brace_preserve_inline = True
            else:
                self.brace_style = bs
        self.unindent_chained_methods = self._get_boolean('unindent_chained_methods')
        self.break_chained_methods = self._get_boolean('break_chained_methods')
        self.space_in_paren = self._get_boolean('space_in_paren')
        self.space_in_empty_paren = self._get_boolean('space_in_empty_paren')
        self.jslint_happy = self._get_boolean('jslint_happy')
        self.space_after_anon_function = self._get_boolean('space_after_anon_function')
        self.space_after_named_function = self._get_boolean('space_after_named_function')
        self.keep_array_indentation = self._get_boolean('keep_array_indentation')
        self.space_before_conditional = self._get_boolean('space_before_conditional', True)
        self.unescape_strings = self._get_boolean('unescape_strings')
        self.e4x = self._get_boolean('e4x')
        self.comma_first = self._get_boolean('comma_first')
        self.operator_position = self._get_selection('operator_position', OPERATOR_POSITION)
        self.test_output_raw = False
        if self.jslint_happy:
            self.space_after_anon_function = True
        self.keep_quiet = False
        self.eval_code = False