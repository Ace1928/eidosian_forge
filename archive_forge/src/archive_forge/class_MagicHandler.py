from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
class MagicHandler(PrefilterHandler):
    handler_name = Unicode('magic')
    esc_strings = List([ESC_MAGIC])

    def handle(self, line_info):
        """Execute magic functions."""
        ifun = line_info.ifun
        the_rest = line_info.the_rest
        t_arg_s = ifun + ' ' + the_rest
        t_magic_name, _, t_magic_arg_s = t_arg_s.partition(' ')
        t_magic_name = t_magic_name.lstrip(ESC_MAGIC)
        cmd = '%sget_ipython().run_line_magic(%r, %r)' % (line_info.pre_whitespace, t_magic_name, t_magic_arg_s)
        return cmd