import abc
import contextlib
import copy
import hashlib
import os
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient import exceptions
from manilaclient import utils
def _get_subresource_and_resource(self, superresource):
    resource = self
    subresource = None
    superresource = superresource or self.superresource
    if superresource is not None:
        resource = superresource
        subresource = self
    return (resource, subresource)