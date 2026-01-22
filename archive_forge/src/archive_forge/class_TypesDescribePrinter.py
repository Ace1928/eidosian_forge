from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict
from googlecloudsdk.core.resource import custom_printer_base as cp
from surface.run.integrations.types.describe import Params
class TypesDescribePrinter(cp.CustomPrinterBase):
    """Prints the types describe block into a custom human-readable format.

  Example output:
    This is an example description of the integration type.

    Parameters:
      param1 [required]:
        Description of param1.

      param2 [optional]:
        Description of param2.

    Example Usage:
      $ gcloud run integrations types create --type=<TYPE>
  """

    def Transform(self, record: Dict[str, str]) -> cp.Lines:
        """Converts the record into a custom format.

    Args:
      record: dict, contains the keys: 'description', 'example_command', and
        'parameters'.

    Returns:
      custom_printer_base.Lines, formatted output for types describe command.
    """
        lines = [record['description'], ' ', cp.Labeled([cp.Lines(['Parameters', self._FormatParams(record['parameters'])])]), cp.Labeled([cp.Lines(['Example Usage', cp.Lines([record['example_command']])])])]
        return cp.Lines(lines)

    def _FormatParams(self, params: Params) -> cp.Lines:
        """Formats all the required and optional parameters for an integration.

    Required parameters should come before optional parameters as defined
    in the PRD.

    Args:
      params: Class contains a list of required and optional params.

    Returns:
      custom_printer_base.Lines, formatted output of all the parameters.
    """
        formatted = []
        for param in params.required:
            formatted.append(self._FormatParam(param, 'required'))
        for param in params.optional:
            formatted.append(self._FormatParam(param, 'optional'))
        return cp.Lines(formatted)

    def _FormatParam(self, param: Dict[str, str], setting: str) -> cp.Labeled:
        """Formats individual parameter for an integration.

    Example output:
      param1 [required]:
        This is a description of param1.

    Args:
      param: contains keys such as 'name' and 'description'
      setting: is either 'required' or 'optional'

    Returns:
      custom_printer_base.Lines, formatted output of a singular parameter.
    """
        return cp.Labeled([cp.Lines(['{} [{}]'.format(param['name'], setting), cp.Lines([param['description'], ' '])])])