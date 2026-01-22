import logging
from oslo_utils import encodeutils
from glance_store import exceptions
from glance_store.i18n import _, _LI
def _get_storage_url(self):
    return self.location.swift_url