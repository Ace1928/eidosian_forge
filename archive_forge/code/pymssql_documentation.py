import re
from .base import MSDialect
from .base import MSIdentifierPreparer
from ... import types as sqltypes
from ... import util
from ...engine import processors

.. dialect:: mssql+pymssql
    :name: pymssql
    :dbapi: pymssql
    :connectstring: mssql+pymssql://<username>:<password>@<freetds_name>/?charset=utf8

pymssql is a Python module that provides a Python DBAPI interface around
`FreeTDS <https://www.freetds.org/>`_.

.. versionchanged:: 2.0.5

    pymssql was restored to SQLAlchemy's continuous integration testing


