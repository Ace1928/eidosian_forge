import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
class MaybeAsyncCommandCompiler(CommandCompiler):

    def __init__(self, extra_flags=0):
        self.compiler = MaybeAsyncCompile(extra_flags=extra_flags)