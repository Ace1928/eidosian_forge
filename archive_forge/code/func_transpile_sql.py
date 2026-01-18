from logging import Logger
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from uuid import uuid4
from triad import to_uuid
from fugue._utils.registry import fugue_plugin
from fugue._utils.misc import import_fsql_dependency
@fugue_plugin
def transpile_sql(raw: str, from_dialect: Optional[str], to_dialect: Optional[str]) -> str:
    """Transpile SQL between dialects, it should work only when both
    ``from_dialect`` and ``to_dialect`` are not None

    :param raw: the raw SQL
    :param from_dialect: the dialect of the raw SQL
    :param to_dialect: the expected dialect.
    :return: the transpiled SQL
    """
    if from_dialect is not None and to_dialect is not None and (from_dialect != to_dialect):
        sqlglot = import_fsql_dependency('sqlglot')
        return ' '.join(sqlglot.transpile(raw, read=from_dialect, write=to_dialect))
    else:
        return raw