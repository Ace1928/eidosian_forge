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
def _MatchPostfix(postfix_props, index_props):
    """Matches a postfix constraint with an existing index.

  postfix_props constraints are specified through a list of:
  - sets of string: any order any direction;
  - list of tuples(string, direction): the given order, and, if specified, the
  given direction.

  For example (PropertySpec objects shown here in their legacy shorthand form):
    [set('A', 'B'), [('C', None), ('D', ASC)]]
  matches:
    [('F', ASC), ('B', ASC), ('A', DESC), ('C', DESC), ('D', ASC)]
  with a return value of [('F', ASC)], but does not match:
    [('F', ASC), ('A', DESC), ('C', DESC), ('D', ASC)]
    [('B', ASC), ('F', ASC), ('A', DESC), ('C', DESC), ('D', ASC)]
    [('F', ASC), ('B', ASC), ('A', DESC), ('C', DESC), ('D', DESC)]

  Args:
    postfix_props: A tuple of sets and lists, as output by
        CompositeIndexForQuery. They should define the requirements for the
        postfix of the index.
    index_props: A list of PropertySpec objects that
        define the index to try and match.

  Returns:
    The list of PropertySpec objects that define the prefix properties
    in the given index.  None if the constraints could not be
    satisfied.

  """
    index_props_rev = reversed(index_props)
    for property_group in reversed(postfix_props):
        index_group_iter = itertools.islice(index_props_rev, len(property_group))
        if isinstance(property_group, (frozenset, set)):
            index_group = set((prop.name for prop in index_group_iter))
            if index_group != property_group:
                return None
        else:
            index_group = list(index_group_iter)
            if len(index_group) != len(property_group):
                return None
            for candidate, spec in zip(index_group, reversed(property_group)):
                if not candidate.Satisfies(spec):
                    return None
    remaining = list(index_props_rev)
    remaining.reverse()
    return remaining