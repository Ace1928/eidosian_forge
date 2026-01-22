from pprint import pformat
from six import iteritems
import re
@additional_printer_columns.setter
def additional_printer_columns(self, additional_printer_columns):
    """
        Sets the additional_printer_columns of this
        V1beta1CustomResourceDefinitionSpec.
        AdditionalPrinterColumns are additional columns shown e.g. in kubectl
        next to the name. Defaults to a created-at column. Optional, the global
        columns for all versions. Top-level and per-version columns are mutually
        exclusive.

        :param additional_printer_columns: The additional_printer_columns of
        this V1beta1CustomResourceDefinitionSpec.
        :type: list[V1beta1CustomResourceColumnDefinition]
        """
    self._additional_printer_columns = additional_printer_columns