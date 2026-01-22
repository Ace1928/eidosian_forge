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
def RemoveNativelySupportedComponents(filters, orders, exists):
    """ Removes query components that are natively supported by the datastore.

  The resulting filters and orders should not be used in an actual query.

  Args:
    filters: the filters set on the query
    orders: the orders set on the query
    exists: the names of properties that require an exists filter if
      not already specified

  Returns:
    (filters, orders) the reduced set of filters and orders
  """
    filters, orders = Normalize(filters, orders, exists)
    for f in filters:
        if f.op() in EXISTS_OPERATORS:
            return (filters, orders)
    has_key_desc_order = False
    if orders and orders[-1].property() == datastore_types.KEY_SPECIAL_PROPERTY:
        if orders[-1].direction() == ASCENDING:
            orders = orders[:-1]
        else:
            has_key_desc_order = True
    if not has_key_desc_order:
        for f in filters:
            if f.op() in INEQUALITY_OPERATORS and f.property(0).name() != datastore_types.KEY_SPECIAL_PROPERTY:
                break
        else:
            filters = [f for f in filters if f.property(0).name() != datastore_types.KEY_SPECIAL_PROPERTY]
    return (filters, orders)