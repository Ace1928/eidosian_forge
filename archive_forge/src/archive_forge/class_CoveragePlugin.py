from __future__ import annotations
import functools
from types import FrameType
from typing import Any, Iterable
from coverage import files
from coverage.misc import _needs_to_implement
from coverage.types import TArc, TConfigurable, TLineNo, TSourceTokenLines
class CoveragePlugin:
    """Base class for coverage.py plug-ins."""
    _coverage_plugin_name: str
    _coverage_enabled: bool

    def file_tracer(self, filename: str) -> FileTracer | None:
        """Get a :class:`FileTracer` object for a file.

        Plug-in type: file tracer.

        Every Python source file is offered to your plug-in to give it a chance
        to take responsibility for tracing the file.  If your plug-in can
        handle the file, it should return a :class:`FileTracer` object.
        Otherwise return None.

        There is no way to register your plug-in for particular files.
        Instead, this method is invoked for all  files as they are executed,
        and the plug-in decides whether it can trace the file or not.
        Be prepared for `filename` to refer to all kinds of files that have
        nothing to do with your plug-in.

        The file name will be a Python file being executed.  There are two
        broad categories of behavior for a plug-in, depending on the kind of
        files your plug-in supports:

        * Static file names: each of your original source files has been
          converted into a distinct Python file.  Your plug-in is invoked with
          the Python file name, and it maps it back to its original source
          file.

        * Dynamic file names: all of your source files are executed by the same
          Python file.  In this case, your plug-in implements
          :meth:`FileTracer.dynamic_source_filename` to provide the actual
          source file for each execution frame.

        `filename` is a string, the path to the file being considered.  This is
        the absolute real path to the file.  If you are comparing to other
        paths, be sure to take this into account.

        Returns a :class:`FileTracer` object to use to trace `filename`, or
        None if this plug-in cannot trace this file.

        """
        return None

    def file_reporter(self, filename: str) -> FileReporter | str:
        """Get the :class:`FileReporter` class to use for a file.

        Plug-in type: file tracer.

        This will only be invoked if `filename` returns non-None from
        :meth:`file_tracer`.  It's an error to return None from this method.

        Returns a :class:`FileReporter` object to use to report on `filename`,
        or the string `"python"` to have coverage.py treat the file as Python.

        """
        _needs_to_implement(self, 'file_reporter')

    def dynamic_context(self, frame: FrameType) -> str | None:
        """Get the dynamically computed context label for `frame`.

        Plug-in type: dynamic context.

        This method is invoked for each frame when outside of a dynamic
        context, to see if a new dynamic context should be started.  If it
        returns a string, a new context label is set for this and deeper
        frames.  The dynamic context ends when this frame returns.

        Returns a string to start a new dynamic context, or None if no new
        context should be started.

        """
        return None

    def find_executable_files(self, src_dir: str) -> Iterable[str]:
        """Yield all of the executable files in `src_dir`, recursively.

        Plug-in type: file tracer.

        Executability is a plug-in-specific property, but generally means files
        which would have been considered for coverage analysis, had they been
        included automatically.

        Returns or yields a sequence of strings, the paths to files that could
        have been executed, including files that had been executed.

        """
        return []

    def configure(self, config: TConfigurable) -> None:
        """Modify the configuration of coverage.py.

        Plug-in type: configurer.

        This method is called during coverage.py start-up, to give your plug-in
        a chance to change the configuration.  The `config` parameter is an
        object with :meth:`~coverage.Coverage.get_option` and
        :meth:`~coverage.Coverage.set_option` methods.  Do not call any other
        methods on the `config` object.

        """
        pass

    def sys_info(self) -> Iterable[tuple[str, Any]]:
        """Get a list of information useful for debugging.

        Plug-in type: any.

        This method will be invoked for ``--debug=sys``.  Your
        plug-in can return any information it wants to be displayed.

        Returns a list of pairs: `[(name, value), ...]`.

        """
        return []