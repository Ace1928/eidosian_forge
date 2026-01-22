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
def _check_all_results_provided(self, atom_name, container):
    """Warn if an atom did not provide some of its expected results.

        This may happen if atom returns shorter tuple or list or dict
        without all needed keys. It may also happen if atom returns
        result of wrong type.
        """
    result_mapping = self._result_mappings.get(atom_name)
    if not result_mapping:
        return
    for name, index in result_mapping.items():
        try:
            _item_from(container, index)
        except _EXTRACTION_EXCEPTIONS:
            LOG.warning("Atom '%s' did not supply result with index %r (name '%s')", atom_name, index, name)