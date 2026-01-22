from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import difflib
import io
import re
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.resource import yaml_printer
class ACMDiffPrinter(resource_printer_base.ResourcePrinter):
    """A printer for an ndiff of the first two projection columns.

  A unified diff of the first two projection columns.

  Printer attributes:
    format: The format of the diffed resources. Each resource is converted
      to this format and the diff of the converted resources is displayed.
      The default is 'yaml'.
  """

    def __init__(self, *args, **kwargs):
        super(ACMDiffPrinter, self).__init__(*args, by_columns=True, non_empty_projection_required=True, **kwargs)
        self._print_format = self.attributes.get('format', 'yaml')

    def _Diff(self, old, new):
        """Prints a modified ndiff of formatter output for old and new.

    IngressPolicies:
     ingressFrom:
       sources:
         accessLevel: accessPolicies/123456789/accessLevels/my_level
        -resource: projects/123456789012
        +resource: projects/234567890123
    EgressPolicies:
      +egressTo:
        +operations:
          +actions:
            +action: method_for_all
            +actionType: METHOD
          +serviceName: chemisttest.googleapis.com
        +resources:
          +projects/345678901234
    Args:
      old: The old original resource.
      new: The new changed resource.
    """
        buf_old = io.StringIO()
        printer = self.Printer(self._print_format, out=buf_old)
        printer.PrintSingleRecord(old)
        buf_new = io.StringIO()
        printer = self.Printer(self._print_format, out=buf_new)
        printer.PrintSingleRecord(new)
        lines_old = ''
        lines_new = ''
        if old is not None:
            lines_old = self._FormatYamlPrinterLinesForDryRunDescribe(buf_old.getvalue().split('\n'))
        if new is not None:
            lines_new = self._FormatYamlPrinterLinesForDryRunDescribe(buf_new.getvalue().split('\n'))
        lines_diff = difflib.ndiff(lines_old, lines_new)
        empty_line_pattern = re.compile('^\\s*$')
        empty_config_pattern = re.compile('^(\\+|-)\\s+\\{\\}$')
        for line in lines_diff:
            if line and line[0] != '?' and (not empty_line_pattern.match(line)) and (not empty_config_pattern.match(line)):
                print(line)

    def _AddRecord(self, record, delimit=False):
        """Immediately prints the first two columns of record as a unified diff.

    Records with less than 2 columns are silently ignored.

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

    def _FormatYamlPrinterLinesForDryRunDescribe(self, lines):
        """Tweak yaml printer formatted resources for ACM's dry run describe output.

    Args:
      lines: yaml printer formatted strings

    Returns:
      lines with no '-' prefix for yaml array elements.
    """
        return [line.replace('-', ' ', 1) for line in lines]