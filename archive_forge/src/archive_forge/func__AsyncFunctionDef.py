import ast
import io
import sys
import tokenize
def _AsyncFunctionDef(self, t):
    self.__FunctionDef_helper(t, 'async def')