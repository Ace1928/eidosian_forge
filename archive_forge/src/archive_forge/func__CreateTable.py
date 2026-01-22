from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import persistent_cache_base
import six
def _CreateTable(self, name, restricted, columns, keys, timeout):
    """Creates and returns a table object for name.

    NOTE: This code is conditioned on self._metadata. If self._metadata is None
    then we are initializing/updating the metadata table. The table name is
    relaxed, in particular '_' is allowed in the table name. This avoids user
    table name conflicts. Finally, self._metadata is set and the metadata
    table row is updated to reflect any changes in the default timeout.

    Args:
      name: The table name.
      restricted: Return a restricted table object.
      columns: The number of columns in each row.
      keys: The number of columns, left to right, that are primary keys. 0 for
        all columns.
      timeout: The number of seconds after last modification when the table
        becomes invalid. 0 for no timeout.

    Raises:
      CacheTableNameInvalid: If name is invalid.

    Returns:
      A table object for name.
    """
    if columns is None:
        columns = 1
    if columns < 1:
        raise exceptions.CacheTableColumnsInvalid('[{}] table [{}] column count [{}] must be >= 1.'.format(self.name, name, columns))
    if keys is None:
        keys = columns
    if keys < 1 or keys > columns:
        raise exceptions.CacheTableKeysInvalid('[{}] table [{}] primary key count [{}] must be >= 1 and <= {}.'.format(self.name, name, keys, columns))
    if timeout is None:
        timeout = self.timeout
    self._ImplementationCreateTable(name, columns, keys)
    table = self._table_class(self, name=name, columns=columns, keys=keys, timeout=timeout, modified=0, restricted=restricted)
    if self._metadata:
        version = None
    else:
        self._metadata = table
        table.Validate()
        rows = table.Select(Metadata.Row(name=name))
        row = rows[0] if rows else None
        if row:
            metadata = Metadata(row)
            if self.version is None:
                self.version = metadata.version or ''
            elif self.version != metadata.version:
                raise exceptions.CacheVersionMismatch('[{}] cache version [{}] does not match [{}].'.format(self.name, metadata.version, self.version), metadata.version, self.version)
            if self.timeout is None:
                self.timeout = metadata.timeout
        table.modified = 0
        version = self.version
    self._metadata.AddRows([Metadata.Row(name=table.name, columns=table.columns, keys=table.keys, timeout=table.timeout, modified=table.modified, restricted=table.restricted, version=version)])
    return table