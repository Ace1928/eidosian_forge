from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def add_insert_auto_id(self):
    x = googlecloudsdk.third_party.appengine.datastore.entity_v4_pb.Entity()
    self.insert_auto_id_.append(x)
    return x