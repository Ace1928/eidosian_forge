from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
class AutoHandler(PrefilterHandler):
    handler_name = Unicode('auto')
    esc_strings = List([ESC_PAREN, ESC_QUOTE, ESC_QUOTE2])

    def handle(self, line_info):
        """Handle lines which can be auto-executed, quoting if requested."""
        line = line_info.line
        ifun = line_info.ifun
        the_rest = line_info.the_rest
        esc = line_info.esc
        continue_prompt = line_info.continue_prompt
        obj = line_info.ofind(self.shell).obj
        if continue_prompt:
            return line
        force_auto = isinstance(obj, IPyAutocall)
        try:
            auto_rewrite = obj.rewrite
        except Exception:
            auto_rewrite = True
        if esc == ESC_QUOTE:
            newcmd = '%s("%s")' % (ifun, '", "'.join(the_rest.split()))
        elif esc == ESC_QUOTE2:
            newcmd = '%s("%s")' % (ifun, the_rest)
        elif esc == ESC_PAREN:
            newcmd = '%s(%s)' % (ifun, ','.join(the_rest.split()))
        else:
            if force_auto:
                do_rewrite = not the_rest.startswith('(')
            elif not the_rest:
                do_rewrite = self.shell.autocall >= 2
            elif the_rest.startswith('[') and hasattr(obj, '__getitem__'):
                do_rewrite = False
            else:
                do_rewrite = True
            if do_rewrite:
                if the_rest.endswith(';'):
                    newcmd = '%s(%s);' % (ifun.rstrip(), the_rest[:-1])
                else:
                    newcmd = '%s(%s)' % (ifun.rstrip(), the_rest)
            else:
                normal_handler = self.prefilter_manager.get_handler_by_name('normal')
                return normal_handler.handle(line_info)
        if auto_rewrite:
            self.shell.auto_rewrite_input(newcmd)
        return newcmd