import threading
def _validate_group_id(self, group_id):
    if group_id < 0 or group_id >= self._num_groups:
        raise ValueError(f'Argument `group_id` should verify `0 <= group_id < num_groups` (with `num_groups={self._num_groups}`). Received: group_id={group_id}')