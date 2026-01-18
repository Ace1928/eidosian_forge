from __future__ import annotations
from collections import ChainMap
from copy import deepcopy
from datetime import timedelta
from typing import TYPE_CHECKING, cast
from streamlit.connections import BaseConnection
from streamlit.connections.util import extract_from_dict
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.caching import cache_data
Return a SQLAlchemy Session.

        Users of this connection should use the contextmanager pattern for writes,
        transactions, and anything more complex than simple read queries.

        See the usage example below, which assumes we have a table ``numbers`` with a
        single integer column ``val``. The `SQLAlchemy
        <https://docs.sqlalchemy.org/en/20/orm/session_basics.html>`_ docs also contain
        much more information on the usage of sessions.

        Example
        -------
        >>> import streamlit as st
        >>> conn = st.connection("sql")
        >>> n = st.slider("Pick a number")
        >>> if st.button("Add the number!"):
        ...     with conn.session as session:
        ...         session.execute("INSERT INTO numbers (val) VALUES (:n);", {"n": n})
        ...         session.commit()
        