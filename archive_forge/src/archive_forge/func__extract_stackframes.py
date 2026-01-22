from os import path
import sys
import traceback
from cupy.cuda import memory_hook
def _extract_stackframes(self):
    stackframes = traceback.extract_stack()
    stackframes = [StackFrame(st) for st in stackframes]
    stackframes = [st for st in stackframes if st.filename != self._filename]
    return stackframes