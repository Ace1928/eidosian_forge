from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
class AutoMagicChecker(PrefilterChecker):
    priority = Integer(700).tag(config=True)

    def check(self, line_info):
        """If the ifun is magic, and automagic is on, run it.  Note: normal,
        non-auto magic would already have been triggered via '%' in
        check_esc_chars. This just checks for automagic.  Also, before
        triggering the magic handler, make sure that there is nothing in the
        user namespace which could shadow it."""
        if not self.shell.automagic or not self.shell.find_magic(line_info.ifun):
            return None
        if line_info.continue_prompt and (not self.prefilter_manager.multi_line_specials):
            return None
        head = line_info.ifun.split('.', 1)[0]
        if is_shadowed(head, self.shell):
            return None
        return self.prefilter_manager.get_handler_by_name('magic')