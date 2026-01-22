from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
class PrefilterHandler(Configurable):
    handler_name = Unicode('normal')
    esc_strings: List = List([])
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)
    prefilter_manager = Instance('IPython.core.prefilter.PrefilterManager', allow_none=True)

    def __init__(self, shell=None, prefilter_manager=None, **kwargs):
        super(PrefilterHandler, self).__init__(shell=shell, prefilter_manager=prefilter_manager, **kwargs)
        self.prefilter_manager.register_handler(self.handler_name, self, self.esc_strings)

    def handle(self, line_info):
        """Handle normal input lines. Use as a template for handlers."""
        line = line_info.line
        continue_prompt = line_info.continue_prompt
        if continue_prompt and self.shell.autoindent and line.isspace() and (0 < abs(len(line) - self.shell.indent_current_nsp) <= 2):
            line = ''
        return line

    def __str__(self):
        return '<%s(name=%s)>' % (self.__class__.__name__, self.handler_name)