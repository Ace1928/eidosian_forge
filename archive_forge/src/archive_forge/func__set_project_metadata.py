import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _set_project_metadata(self, metadata=None, force=False, current_keys=''):
    """
        Return the GCE-friendly dictionary of metadata with/without an
        entry for 'sshKeys' based on params for 'force' and 'current_keys'.
        This method was added to simplify the set_common_instance_metadata
        method and make it easier to test.

        :param  metadata: The GCE-formatted dict (e.g. 'items' list of dicts)
        :type   metadata: ``dict`` or ``None``

        :param  force: Flag to specify user preference for keeping current_keys
        :type   force: ``bool``

        :param  current_keys: The value, if any, of existing 'sshKeys'
        :type   current_keys: ``str``

        :return: GCE-friendly metadata dict
        :rtype:  ``dict``
        """
    if metadata is None:
        if not force and current_keys:
            new_md = [{'key': 'sshKeys', 'value': current_keys}]
        else:
            new_md = []
    else:
        new_md = metadata['items']
        if not force and current_keys:
            updated_md = []
            for d in new_md:
                if d['key'] != 'sshKeys':
                    updated_md.append({'key': d['key'], 'value': d['value']})
            new_md = updated_md
            new_md.append({'key': 'sshKeys', 'value': current_keys})
    return new_md