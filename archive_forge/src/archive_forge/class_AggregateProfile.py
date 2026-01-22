import os
class AggregateProfile(object):
    """Profile summary data for aggregating a number of ProfileDatum."""

    def __init__(self, profile_datum):
        """Constructor.

    Args:
      profile_datum: (`ProfileDatum`) an instance of `ProfileDatum` to
        initialize this object with.
    """
        self.total_op_time = profile_datum.op_time
        self.total_exec_time = profile_datum.exec_time
        device_and_node = '%s:%s' % (profile_datum.device_name, profile_datum.node_exec_stats.node_name)
        self._node_to_exec_count = {device_and_node: 1}

    def add(self, profile_datum):
        """Accumulate a new instance of ProfileDatum.

    Args:
      profile_datum: (`ProfileDatum`) an instance of `ProfileDatum` to
        accumulate to this object.
    """
        self.total_op_time += profile_datum.op_time
        self.total_exec_time += profile_datum.exec_time
        device_and_node = '%s:%s' % (profile_datum.device_name, profile_datum.node_exec_stats.node_name)
        device_and_node = '%s:%s' % (profile_datum.device_name, profile_datum.node_exec_stats.node_name)
        if device_and_node in self._node_to_exec_count:
            self._node_to_exec_count[device_and_node] += 1
        else:
            self._node_to_exec_count[device_and_node] = 1

    @property
    def node_count(self):
        return len(self._node_to_exec_count)

    @property
    def node_exec_count(self):
        return sum(self._node_to_exec_count.values())