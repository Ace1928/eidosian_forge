from __future__ import annotations
from datetime import timedelta
from typing import TYPE_CHECKING, cast
from streamlit.connections import BaseConnection
from streamlit.connections.util import running_in_sis
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.caching import cache_data
def write_pandas(self, df: DataFrame, table_name: str, database: str | None=None, schema: str | None=None, chunk_size: int | None=None, **kwargs) -> tuple[bool, int, int]:
    """Call snowflake.connector.pandas_tools.write_pandas with this connection.

        This convenience method is simply a thin wrapper around the ``write_pandas``
        function of the same name from ``snowflake.connector.pandas_tools``. For more
        information, see the `Snowflake Python Connector documentation
        <https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api#write_pandas>`_.

        Returns
        -------
        tuple[bool, int, int]
            A tuple containing three values:
                1. A bool that is True if the write was successful.
                2. An int giving the number of chunks of data that were copied.
                3. An int giving the number of rows that were inserted.
        """
    from snowflake.connector.pandas_tools import write_pandas
    success, nchunks, nrows, _ = write_pandas(conn=self._instance, df=df, table_name=table_name, database=database, schema=schema, chunk_size=chunk_size, **kwargs)
    return (success, nchunks, nrows)