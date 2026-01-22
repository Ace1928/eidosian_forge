import os
import sys
import tempfile
from IPython.core.compilerop import CachingCompiler
class XCachingCompiler(CachingCompiler):
    """A custom caching compiler."""

    def __init__(self, *args, **kwargs):
        """Initialize the compiler."""
        super().__init__(*args, **kwargs)
        self.log = None

    def get_code_name(self, raw_code, code, number):
        """Get the code name."""
        return get_file_name(raw_code)