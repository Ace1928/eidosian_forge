from __future__ import annotations
import collections.abc as collections_abc
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import quote_plus
from urllib.parse import unquote
from .interfaces import Dialect
from .. import exc
from .. import util
from ..dialects import plugins
from ..dialects import registry
def update_query_string(self, query_string: str, append: bool=False) -> URL:
    """Return a new :class:`_engine.URL` object with the :attr:`_engine.URL.query`
        parameter dictionary updated by the given query string.

        E.g.::

            >>> from sqlalchemy.engine import make_url
            >>> url = make_url("postgresql+psycopg2://user:pass@host/dbname")
            >>> url = url.update_query_string("alt_host=host1&alt_host=host2&ssl_cipher=%2Fpath%2Fto%2Fcrt")
            >>> str(url)
            'postgresql+psycopg2://user:pass@host/dbname?alt_host=host1&alt_host=host2&ssl_cipher=%2Fpath%2Fto%2Fcrt'

        :param query_string: a URL escaped query string, not including the
         question mark.

        :param append: if True, parameters in the existing query string will
         not be removed; new parameters will be in addition to those present.
         If left at its default of False, keys present in the given query
         parameters will replace those of the existing query string.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_engine.URL.query`

            :meth:`_engine.URL.update_query_dict`

        """
    return self.update_query_pairs(parse_qsl(query_string), append=append)