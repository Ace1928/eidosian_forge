from __future__ import annotations
import functools
from types import FrameType
from typing import Any, Iterable
from coverage import files
from coverage.misc import _needs_to_implement
from coverage.types import TArc, TConfigurable, TLineNo, TSourceTokenLines
class FileTracer(CoveragePluginBase):
    """Support needed for files during the execution phase.

    File tracer plug-ins implement subclasses of FileTracer to return from
    their :meth:`~CoveragePlugin.file_tracer` method.

    You may construct this object from :meth:`CoveragePlugin.file_tracer` any
    way you like.  A natural choice would be to pass the file name given to
    `file_tracer`.

    `FileTracer` objects should only be created in the
    :meth:`CoveragePlugin.file_tracer` method.

    See :ref:`howitworks` for details of the different coverage.py phases.

    """

    def source_filename(self) -> str:
        """The source file name for this file.

        This may be any file name you like.  A key responsibility of a plug-in
        is to own the mapping from Python execution back to whatever source
        file name was originally the source of the code.

        See :meth:`CoveragePlugin.file_tracer` for details about static and
        dynamic file names.

        Returns the file name to credit with this execution.

        """
        _needs_to_implement(self, 'source_filename')

    def has_dynamic_source_filename(self) -> bool:
        """Does this FileTracer have dynamic source file names?

        FileTracers can provide dynamically determined file names by
        implementing :meth:`dynamic_source_filename`.  Invoking that function
        is expensive. To determine whether to invoke it, coverage.py uses the
        result of this function to know if it needs to bother invoking
        :meth:`dynamic_source_filename`.

        See :meth:`CoveragePlugin.file_tracer` for details about static and
        dynamic file names.

        Returns True if :meth:`dynamic_source_filename` should be called to get
        dynamic source file names.

        """
        return False

    def dynamic_source_filename(self, filename: str, frame: FrameType) -> str | None:
        """Get a dynamically computed source file name.

        Some plug-ins need to compute the source file name dynamically for each
        frame.

        This function will not be invoked if
        :meth:`has_dynamic_source_filename` returns False.

        Returns the source file name for this frame, or None if this frame
        shouldn't be measured.

        """
        return None

    def line_number_range(self, frame: FrameType) -> tuple[TLineNo, TLineNo]:
        """Get the range of source line numbers for a given a call frame.

        The call frame is examined, and the source line number in the original
        file is returned.  The return value is a pair of numbers, the starting
        line number and the ending line number, both inclusive.  For example,
        returning (5, 7) means that lines 5, 6, and 7 should be considered
        executed.

        This function might decide that the frame doesn't indicate any lines
        from the source file were executed.  Return (-1, -1) in this case to
        tell coverage.py that no lines should be recorded for this frame.

        """
        lineno = frame.f_lineno
        return (lineno, lineno)