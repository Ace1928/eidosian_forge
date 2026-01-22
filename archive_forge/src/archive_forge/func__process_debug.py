import re
import ast
from hacking import core
def _process_debug(self, node):
    msg = node.args[0]
    if isinstance(msg, ast.Call) and isinstance(msg.func, ast.Name):
        self.add_error(msg, message=self.DEBUG_CHECK_DESC)
    elif isinstance(msg, ast.Name) and msg.id in self.assignments and (not self._is_raised_later(node, msg.id)):
        self.add_error(msg, message=self.DEBUG_CHECK_DESC)