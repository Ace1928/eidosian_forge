from typing import Any, Dict, Optional, Sequence
class ModinDatabaseConnection:
    """
    Creates a SQL database connection.

    Parameters
    ----------
    lib : str
        The library for the SQL connection.
    *args : iterable
        Positional arguments to pass when creating the connection.
    **kwargs : dict
        Keyword arguments to pass when creating the connection.
    """
    lib: str
    args: Sequence
    kwargs: Dict
    _dialect_is_microsoft_sql_cache: Optional[bool]

    def __init__(self, lib: str, *args: Any, **kwargs: Any) -> None:
        lib = lib.lower()
        if lib not in (_PSYCOPG_LIB_NAME, _SQLALCHEMY_LIB_NAME):
            raise UnsupportedDatabaseException(f'Unsupported database library {lib}')
        self.lib = lib
        self.args = args
        self.kwargs = kwargs
        self._dialect_is_microsoft_sql_cache = None

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

    def get_connection(self) -> Any:
        """
        Make the database connection and get it.

        For psycopg2, pass all arguments to psycopg2.connect() and return the
        result of psycopg2.connect(). For sqlalchemy, pass all arguments to
        sqlalchemy.create_engine() and return the result of calling connect()
        on the engine.

        Returns
        -------
        Any
            The open database connection.
        """
        if self.lib == _PSYCOPG_LIB_NAME:
            import psycopg2
            return psycopg2.connect(*self.args, **self.kwargs)
        if self.lib == _SQLALCHEMY_LIB_NAME:
            from sqlalchemy import create_engine
            return create_engine(*self.args, **self.kwargs).connect()
        raise UnsupportedDatabaseException('Unsupported database library')

    def get_string(self) -> str:
        """
        Get input connection string.

        Returns
        -------
        str
        """
        return self.args[0]

    def column_names_query(self, query: str) -> str:
        """
        Get a query that gives the names of columns that `query` would produce.

        Parameters
        ----------
        query : str
            The SQL query to check.

        Returns
        -------
        str
        """
        return f'SELECT * FROM ({query}) AS _MODIN_COUNT_QUERY WHERE 1 = 0'

    def row_count_query(self, query: str) -> str:
        """
        Get a query that gives the names of rows that `query` would produce.

        Parameters
        ----------
        query : str
            The SQL query to check.

        Returns
        -------
        str
        """
        return f'SELECT COUNT(*) FROM ({query}) AS _MODIN_COUNT_QUERY'

    def partition_query(self, query: str, limit: int, offset: int) -> str:
        """
        Get a query that partitions the original `query`.

        Parameters
        ----------
        query : str
            The SQL query to get a partition.
        limit : int
            The size of the partition.
        offset : int
            Where the partition begins.

        Returns
        -------
        str
        """
        return f'SELECT * FROM ({query}) AS _MODIN_COUNT_QUERY ORDER BY(SELECT NULL)' + f' OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY' if self._dialect_is_microsoft_sql() else f'SELECT * FROM ({query}) AS _MODIN_COUNT_QUERY LIMIT ' + f'{limit} OFFSET {offset}'