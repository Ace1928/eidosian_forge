from __future__ import absolute_import
from ruamel import yaml
import copy
import itertools
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_object
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
def PropertyPresenter(dumper, prop):
    """A PyYaml presenter for Property.

  It differs from the default by not outputting 'mode: null' and direction when
  mode is specified. This is done in order to ensure backwards compatibility.

  Args:
    dumper: the Dumper object provided by PyYaml.
    prop: the Property object to serialize.

  Returns:
    A PyYaml object mapping.
  """
    prop_copy = copy.copy(prop)
    if prop.mode is None:
        del prop_copy.mode
    if prop.direction is None:
        del prop_copy.direction
    return dumper.represent_object(prop_copy)