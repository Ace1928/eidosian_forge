import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
def _item_from_first_of(providers, looking_for):
    """Returns item from the *first* successful container extraction."""
    for provider, container in providers:
        try:
            return (provider, _item_from(container, provider.index))
        except _EXTRACTION_EXCEPTIONS:
            pass
    providers = [p[0] for p in providers]
    raise exceptions.NotFound('Unable to find result %r, expected to be able to find it created by one of %s but was unable to perform successful extraction' % (looking_for, providers))