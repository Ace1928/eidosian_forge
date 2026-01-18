import decimal
import re
from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .json import JSON
from .json import JSONB
from .json import JSONPathType
from .pg_catalog import _SpaceVector
from .pg_catalog import OIDVECTOR
from .types import CITEXT
from ... import exc
from ... import util
from ...engine import processors
from ...sql import sqltypes
from ...sql.elements import quoted_name

.. dialect:: postgresql+pg8000
    :name: pg8000
    :dbapi: pg8000
    :connectstring: postgresql+pg8000://user:password@host:port/dbname[?key=value&key=value...]
    :url: https://pypi.org/project/pg8000/

.. versionchanged:: 1.4  The pg8000 dialect has been updated for version
   1.16.6 and higher, and is again part of SQLAlchemy's continuous integration
   with full feature support.

.. _pg8000_unicode:

Unicode
-------

pg8000 will encode / decode string values between it and the server using the
PostgreSQL ``client_encoding`` parameter; by default this is the value in
the ``postgresql.conf`` file, which often defaults to ``SQL_ASCII``.
Typically, this can be changed to ``utf-8``, as a more useful default::

    #client_encoding = sql_ascii # actually, defaults to database
                                 # encoding
    client_encoding = utf8

The ``client_encoding`` can be overridden for a session by executing the SQL:

SET CLIENT_ENCODING TO 'utf8';

SQLAlchemy will execute this SQL on all new connections based on the value
passed to :func:`_sa.create_engine` using the ``client_encoding`` parameter::

    engine = create_engine(
        "postgresql+pg8000://user:pass@host/dbname", client_encoding='utf8')

.. _pg8000_ssl:

SSL Connections
---------------

pg8000 accepts a Python ``SSLContext`` object which may be specified using the
:paramref:`_sa.create_engine.connect_args` dictionary::

    import ssl
    ssl_context = ssl.create_default_context()
    engine = sa.create_engine(
        "postgresql+pg8000://scott:tiger@192.168.0.199/test",
        connect_args={"ssl_context": ssl_context},
    )

If the server uses an automatically-generated certificate that is self-signed
or does not match the host name (as seen from the client), it may also be
necessary to disable hostname checking::

    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    engine = sa.create_engine(
        "postgresql+pg8000://scott:tiger@192.168.0.199/test",
        connect_args={"ssl_context": ssl_context},
    )

.. _pg8000_isolation_level:

pg8000 Transaction Isolation Level
-------------------------------------

The pg8000 dialect offers the same isolation level settings as that
of the :ref:`psycopg2 <psycopg2_isolation_level>` dialect:

* ``READ COMMITTED``
* ``READ UNCOMMITTED``
* ``REPEATABLE READ``
* ``SERIALIZABLE``
* ``AUTOCOMMIT``

.. seealso::

    :ref:`postgresql_isolation_level`

    :ref:`psycopg2_isolation_level`


