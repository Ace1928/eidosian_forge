from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import file_cache
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def SelectTable(self, table, row_template, parameter_info, aggregations=None):
    """Returns the list of rows matching row_template in table.

    Refreshes expired tables by calling the updater.

    Args:
      table: The persistent table object.
      row_template: A row template to match in Select().
      parameter_info: A ParamaterInfo object for accessing parameter values in
        the program state.
      aggregations: A list of aggregation Parameter objects.

    Returns:
      The list of rows matching row_template in table.
    """
    if not aggregations:
        aggregations = []
    log.info('cache table=%s aggregations=[%s]', table.name, ' '.join(['{}={}'.format(x.name, x.value) for x in aggregations]))
    try:
        return table.Select(row_template)
    except exceptions.CacheTableExpired:
        rows = self.Update(parameter_info, aggregations)
        if rows is not None:
            table.DeleteRows()
            table.AddRows(rows)
            table.Validate()
        return table.Select(row_template, ignore_expiration=True)