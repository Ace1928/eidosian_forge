import contextlib
from oslo_utils import uuidutils
from taskflow import logging
from taskflow.persistence import models
Creates a flow detail for a flow & adds & saves it in a logbook.

    This will create a flow detail for the given flow using the flow name,
    and add it to the provided logbook and then uses the given backend to save
    the logbook and then returns the created flow detail.

    If no book is provided a temporary one will be created automatically (no
    reference to the logbook will be returned, so this should nearly *always*
    be provided or only used in situations where no logbook is needed, for
    example in tests). If no backend is provided then no saving will occur and
    the created flow detail will not be persisted even if the flow detail was
    added to a given (or temporarily generated) logbook.
    