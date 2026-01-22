from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class LanguageInfo(object):
    """
    This class defines the interface that ExecutionControlCommandBase needs to
    provide language-specific execution control.

    Classes that implement this interface should implement:

        lineno(frame)
            Tells the current line number (only called for a relevant frame).
            If lineno is a false value it is not checked for a difference.

        is_relevant_function(frame)
            tells whether we care about frame 'frame'

        get_source_line(frame)
            get the line of source code for the current line (only called for a
            relevant frame). If the source code cannot be retrieved this
            function should return None

        exc_info(frame) -- optional
            tells whether an exception was raised, if so, it should return a
            string representation of the exception value, None otherwise.

        static_break_functions()
            returns an iterable of function names that are considered relevant
            and should halt step-into execution. This is needed to provide a
            performing step-into

        runtime_break_functions() -- optional
            list of functions that we should break into depending on the
            context
    """

    def exc_info(self, frame):
        """See this class' docstring."""

    def runtime_break_functions(self):
        """
        Implement this if the list of step-into functions depends on the
        context.
        """
        return ()