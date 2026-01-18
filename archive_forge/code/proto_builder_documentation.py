from collections import OrderedDict
import hashlib
import os
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message_factory
Populate FileDescriptorProto for MessageFactory's DescriptorPool.