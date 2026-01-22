from typing import Any, Dict, Optional, Sequence
def _dialect_is_microsoft_sql(self) -> bool:
    """
        Tell whether this connection requires Microsoft SQL dialect.

        If this is a sqlalchemy connection, create an engine from args and
        kwargs. If that engine's driver is pymssql or pyodbc, this
        connection requires Microsoft SQL. Otherwise, it doesn't.

        Returns
        -------
        bool
        """
    if self._dialect_is_microsoft_sql_cache is None:
        self._dialect_is_microsoft_sql_cache = False
        if self.lib == _SQLALCHEMY_LIB_NAME:
            from sqlalchemy import create_engine
            self._dialect_is_microsoft_sql_cache = create_engine(*self.args, **self.kwargs).driver in ('pymssql', 'pyodbc')
    return self._dialect_is_microsoft_sql_cache