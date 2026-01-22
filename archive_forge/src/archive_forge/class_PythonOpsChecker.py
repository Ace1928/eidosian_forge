from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
class PythonOpsChecker(PrefilterChecker):
    priority = Integer(900).tag(config=True)

    def check(self, line_info):
        """If the 'rest' of the line begins with a function call or pretty much
        any python operator, we should simply execute the line (regardless of
        whether or not there's a possible autocall expansion).  This avoids
        spurious (and very confusing) geattr() accesses."""
        if line_info.the_rest and line_info.the_rest[0] in '!=()<>,+*/%^&|':
            return self.prefilter_manager.get_handler_by_name('normal')
        else:
            return None