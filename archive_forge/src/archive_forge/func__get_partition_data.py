import boto.vendored.regions.regions as _regions
def _get_partition_data(self, partition_name):
    """Get partition information for a particular partition.

        This should NOT be used to get service endpoint data because it only
        loads from the new endpoint format. It should only be used for
        partition metadata and partition specific service metadata.

        :type partition_name: str
        :param partition_name: The name of the partition to search for.

        :returns: Partition info from the new endpoints format.
        :rtype: dict or None
        """
    for partition in self._endpoint_data['partitions']:
        if partition['partition'] == partition_name:
            return partition
    raise ValueError('Could not find partition data for: %s' % partition_name)