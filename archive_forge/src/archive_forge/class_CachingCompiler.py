import __future__
from ast import PyCF_ONLY_AST
import codeop
import functools
import hashlib
import linecache
import operator
import time
from contextlib import contextmanager
class CachingCompiler(codeop.Compile):
    """A compiler that caches code compiled from interactive statements.
    """

    def __init__(self):
        codeop.Compile.__init__(self)
        self._filename_map = {}

    def ast_parse(self, source, filename='<unknown>', symbol='exec'):
        """Parse code to an AST with the current compiler flags active.

        Arguments are exactly the same as ast.parse (in the standard library),
        and are passed to the built-in compile function."""
        return compile(source, filename, symbol, self.flags | PyCF_ONLY_AST, 1)

    def reset_compiler_flags(self):
        """Reset compiler flags to default state."""
        self.flags = codeop.PyCF_DONT_IMPLY_DEDENT

    @property
    def compiler_flags(self):
        """Flags currently active in the compilation process.
        """
        return self.flags

    def get_code_name(self, raw_code, transformed_code, number):
        """Compute filename given the code, and the cell number.

        Parameters
        ----------
        raw_code : str
            The raw cell code.
        transformed_code : str
            The executable Python source code to cache and compile.
        number : int
            A number which forms part of the code's name. Used for the execution
            counter.

        Returns
        -------
        The computed filename.
        """
        return code_name(transformed_code, number)

    def format_code_name(self, name):
        """Return a user-friendly label and name for a code block.

        Parameters
        ----------
        name : str
            The name for the code block returned from get_code_name

        Returns
        -------
        A (label, name) pair that can be used in tracebacks, or None if the default formatting should be used.
        """
        if name in self._filename_map:
            return ('Cell', 'In[%s]' % self._filename_map[name])

    def cache(self, transformed_code, number=0, raw_code=None):
        """Make a name for a block of code, and cache the code.

        Parameters
        ----------
        transformed_code : str
            The executable Python source code to cache and compile.
        number : int
            A number which forms part of the code's name. Used for the execution
            counter.
        raw_code : str
            The raw code before transformation, if None, set to `transformed_code`.

        Returns
        -------
        The name of the cached code (as a string). Pass this as the filename
        argument to compilation, so that tracebacks are correctly hooked up.
        """
        if raw_code is None:
            raw_code = transformed_code
        name = self.get_code_name(raw_code, transformed_code, number)
        self._filename_map[name] = number
        entry = (len(transformed_code), None, [line + '\n' for line in transformed_code.splitlines()], name)
        linecache.cache[name] = entry
        return name

    @contextmanager
    def extra_flags(self, flags):
        turn_on_bits = ~self.flags & flags
        self.flags = self.flags | flags
        try:
            yield
        finally:
            self.flags &= ~turn_on_bits