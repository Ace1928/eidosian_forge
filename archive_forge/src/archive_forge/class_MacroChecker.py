from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
class MacroChecker(PrefilterChecker):
    priority = Integer(250).tag(config=True)

    def check(self, line_info):
        obj = self.shell.user_ns.get(line_info.ifun)
        if isinstance(obj, Macro):
            return self.prefilter_manager.get_handler_by_name('macro')
        else:
            return None