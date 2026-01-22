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
def _atomdetail_by_name(self, atom_name, expected_type=None, clone=False):
    try:
        ad = self._flowdetail.find(self._atom_name_to_uuid[atom_name])
    except KeyError:
        exceptions.raise_with_cause(exceptions.NotFound, "Unknown atom name '%s'" % atom_name)
    else:
        if expected_type and (not isinstance(ad, expected_type)):
            raise TypeError("Atom '%s' is not of the expected type: %s" % (atom_name, reflection.get_class_name(expected_type)))
        if clone:
            return (ad, ad.copy())
        else:
            return (ad, ad)