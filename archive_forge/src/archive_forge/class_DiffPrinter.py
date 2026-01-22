from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.core.resource import resource_printer_base
class DiffPrinter(resource_printer_base.ResourcePrinter):
    """A printer for a unified diff of the first two projection columns.

  A unified diff of the first two projection columns.

  Printer attributes:
    format: The format of the diffed resources. Each resource is converted
      to this format and the diff of the converted resources is displayed.
      The default is 'flattened'.
  """

    def __init__(self, *args, **kwargs):
        super(DiffPrinter, self).__init__(*args, by_columns=True, non_empty_projection_required=True, **kwargs)
        self._print_format = self.attributes.get('format', 'flattened')

    def _Diff(self, old, new):
        """Prints the unified diff of formatter output for old and new.

    Prints a unified diff, eg,
    ---

    +++

    @@ -27,6 +27,6 @@

     settings.pricingPlan:                             PER_USE
     settings.replicationType:                         SYNCHRONOUS
     settings.settingsVersion:                         1
    -settings.tier:                                    D1
    +settings.tier:                                    D0
     state:                                            RUNNABLE

    Args:
      old: The old original resource.
      new: The new changed resource.
    """
        import difflib
        buf_old = io.StringIO()
        printer = self.Printer(self._print_format, out=buf_old)
        printer.PrintSingleRecord(old)
        buf_new = io.StringIO()
        printer = self.Printer(self._print_format, out=buf_new)
        printer.PrintSingleRecord(new)
        lines_old = buf_old.getvalue().split('\n')
        lines_new = buf_new.getvalue().split('\n')
        lines_diff = difflib.unified_diff(lines_old, lines_new)
        for line in lines_diff:
            self._out.Print(line)

    def _AddRecord(self, record, delimit=False):
        """Immediately prints the first two columns of record as a unified diff.

    Records with less than 2 colums are silently ignored.

    Args:
      record: A JSON-serializable object.
      delimit: Prints resource delimiters if True.
    """
        title = self.attributes.get('title')
        if title:
            self._out.Print(title)
            self._title = None
        if len(record) > 1:
            self._Diff(record[0], record[1])