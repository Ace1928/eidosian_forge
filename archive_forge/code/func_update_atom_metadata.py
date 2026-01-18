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
def update_atom_metadata(self, atom_name, update_with):
    """Updates a atoms associated metadata.

        This update will take a provided dictionary or a list of (key, value)
        pairs to include in the updated metadata (newer keys will overwrite
        older keys) and after merging saves the updated data into the
        underlying persistence layer.
        """
    self._update_atom_metadata(atom_name, update_with)