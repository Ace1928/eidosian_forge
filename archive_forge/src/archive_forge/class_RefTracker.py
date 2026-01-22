from __future__ import absolute_import
class RefTracker(object):

    def __init__(self):
        self.last_ref = None
        self.last_ids = {}
        self.heads = {}

    def dump_stats(self, note):
        self._show_stats_for(self.last_ids, 'last-ids', note=note)
        self._show_stats_for(self.heads, 'heads', note=note)

    def clear(self):
        self.last_ids.clear()
        self.heads.clear()

    def track_heads(self, cmd):
        """Track the repository heads given a CommitCommand.

        :param cmd: the CommitCommand
        :return: the list of parents in terms of commit-ids
        """
        if cmd.from_ is not None:
            parents = [cmd.from_]
        else:
            last_id = self.last_ids.get(cmd.ref)
            if last_id is not None:
                parents = [last_id]
            else:
                parents = []
        parents.extend(cmd.merges)
        self.track_heads_for_ref(cmd.ref, cmd.id, parents)
        return parents

    def track_heads_for_ref(self, cmd_ref, cmd_id, parents=None):
        if parents is not None:
            for parent in parents:
                if parent in self.heads:
                    del self.heads[parent]
        self.heads.setdefault(cmd_id, set()).add(cmd_ref)
        self.last_ids[cmd_ref] = cmd_id
        self.last_ref = cmd_ref