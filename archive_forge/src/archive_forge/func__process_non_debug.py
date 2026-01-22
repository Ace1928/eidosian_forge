import re
import ast
from hacking import core
def _process_non_debug(self, node, method_name):
    msg = node.args[0]
    if isinstance(msg, ast.Call):
        self.add_error(msg, message=self.NONDEBUG_CHECK_DESC)
    elif isinstance(msg, ast.Name):
        if msg.id not in self.assignments:
            return
        if self._is_raised_later(node, msg.id):
            self.add_error(msg, message=self.NONDEBUG_CHECK_DESC)
        elif self._is_raised_later(node, msg.id):
            self.add_error(msg, message=self.EXCESS_HELPER_CHECK_DESC)