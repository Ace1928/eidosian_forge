from __future__ import absolute_import
def track_heads_for_ref(self, cmd_ref, cmd_id, parents=None):
    if parents is not None:
        for parent in parents:
            if parent in self.heads:
                del self.heads[parent]
    self.heads.setdefault(cmd_id, set()).add(cmd_ref)
    self.last_ids[cmd_ref] = cmd_id
    self.last_ref = cmd_ref