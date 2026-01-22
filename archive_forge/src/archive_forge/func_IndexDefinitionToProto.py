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
def IndexDefinitionToProto(app_id, index_definition):
    """Transform individual Index definition to protocol buffer.

  Args:
    app_id: Application id for new protocol buffer CompositeIndex.
    index_definition: datastore_index.Index object to transform.

  Returns:
    New entity_pb.CompositeIndex with default values set and index
    information filled in.
  """
    proto = entity_pb.CompositeIndex()
    proto.set_app_id(app_id)
    proto.set_id(0)
    proto.set_state(entity_pb.CompositeIndex.WRITE_ONLY)
    definition_proto = proto.mutable_definition()
    definition_proto.set_entity_type(index_definition.kind)
    definition_proto.set_ancestor(index_definition.ancestor)
    if index_definition.properties is not None:
        is_geo = any((x.mode == 'geospatial' for x in index_definition.properties))
        for prop in index_definition.properties:
            prop_proto = definition_proto.add_property()
            prop_proto.set_name(prop.name)
            if prop.mode == 'geospatial':
                prop_proto.set_mode(entity_pb.Index_Property.GEOSPATIAL)
            elif is_geo:
                pass
            elif prop.IsAscending():
                prop_proto.set_direction(entity_pb.Index_Property.ASCENDING)
            else:
                prop_proto.set_direction(entity_pb.Index_Property.DESCENDING)
    return proto