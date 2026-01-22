from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import io
import ipaddress
import os
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def BuildPartialUpdate(clear, remove_keys, set_entries, field_mask_prefix, entry_cls, env_builder):
    """Builds the field mask and patch environment for an environment update.

  Follows the environments update semantic which applies operations
  in an effective order of clear -> remove -> set.

  Leading and trailing whitespace is stripped from elements in remove_keys
  and the keys of set_entries.

  Args:
    clear: bool, If true, the patch removes existing keys.
    remove_keys: iterable(string), Iterable of keys to remove.
    set_entries: {string: string}, Dict containing entries to set.
    field_mask_prefix: string, The prefix defining the path to the base of the
      proto map to be patched.
    entry_cls: AdditionalProperty, The AdditionalProperty class for the type of
      entry being updated.
    env_builder: [AdditionalProperty] -> Environment, A function which produces
      a patch Environment with the given list of entry_cls properties.

  Returns:
    (string, Environment), a 2-tuple of the field mask defined by the arguments
    and a patch environment produced by env_builder.
  """
    remove_keys = set((k.strip() for k in remove_keys or []))
    set_entries = collections.OrderedDict(((k.strip(), v) for k, v in sorted(six.iteritems(set_entries or {}))))
    if clear:
        entries = [entry_cls(key=key, value=value) for key, value in six.iteritems(set_entries)]
        return (field_mask_prefix, env_builder(entries))
    field_mask_entries = []
    seen_keys = set()
    for key in remove_keys:
        field_mask_entries.append('{}.{}'.format(field_mask_prefix, key))
        seen_keys.add(key)
    entries = []
    for key, value in six.iteritems(set_entries):
        entries.append(entry_cls(key=key, value=value))
        if key not in seen_keys:
            field_mask_entries.append('{}.{}'.format(field_mask_prefix, key))
    field_mask_entries.sort()
    return (','.join(field_mask_entries), env_builder(entries))