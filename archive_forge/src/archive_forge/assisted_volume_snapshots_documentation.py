from oslo_serialization import jsonutils
from novaclient import base

        Delete a specified assisted volume snapshot.

        :param snapshot: an assisted volume snapshot to delete
        :param delete_info: Information for snapshot deletion
        :returns: An instance of novaclient.base.TupleWithMeta
        