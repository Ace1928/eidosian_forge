from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
class AutocallChecker(PrefilterChecker):
    priority = Integer(1000).tag(config=True)
    function_name_regexp = CRegExp(re_fun_name, help='RegExp to identify potential function names.').tag(config=True)
    exclude_regexp = CRegExp(re_exclude_auto, help='RegExp to exclude strings with this start from autocalling.').tag(config=True)

    def check(self, line_info):
        """Check if the initial word/function is callable and autocall is on."""
        if not self.shell.autocall:
            return None
        oinfo = line_info.ofind(self.shell)
        if not oinfo.found:
            return None
        ignored_funs = ['b', 'f', 'r', 'u', 'br', 'rb', 'fr', 'rf']
        ifun = line_info.ifun
        line = line_info.line
        if ifun.lower() in ignored_funs and (line.startswith(ifun + "'") or line.startswith(ifun + '"')):
            return None
        if callable(oinfo.obj) and (not self.exclude_regexp.match(line_info.the_rest)) and self.function_name_regexp.match(line_info.ifun):
            return self.prefilter_manager.get_handler_by_name('auto')
        else:
            return None