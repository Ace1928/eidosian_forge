import contextlib
from oslo_utils import importutils
from oslo_utils import reflection
import stevedore.driver
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence import backends as p_backends
from taskflow.utils import misc
from taskflow.utils import persistence_utils as p_utils
def _extract_engine(engine, **kwargs):
    """Extracts the engine kind and any associated options."""
    kind = engine
    if not kind:
        kind = ENGINE_DEFAULT
    options = {}
    try:
        uri = misc.parse_uri(kind)
    except (TypeError, ValueError):
        pass
    else:
        kind = uri.scheme
        options = misc.merge_uri(uri, options.copy())
    options.update(kwargs)
    return (kind, options)