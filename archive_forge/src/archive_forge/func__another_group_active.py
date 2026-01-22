import threading
def _another_group_active(self, group_id):
    return any((c > 0 for g, c in enumerate(self._group_member_counts) if g != group_id))