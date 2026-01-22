from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
class AssignmentChecker(PrefilterChecker):
    priority = Integer(600).tag(config=True)

    def check(self, line_info):
        """Check to see if user is assigning to a var for the first time, in
        which case we want to avoid any sort of automagic / autocall games.

        This allows users to assign to either alias or magic names true python
        variables (the magic/alias systems always take second seat to true
        python code).  E.g. ls='hi', or ls,that=1,2"""
        if line_info.the_rest:
            if line_info.the_rest[0] in '=,':
                return self.prefilter_manager.get_handler_by_name('normal')
        else:
            return None