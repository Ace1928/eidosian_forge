from manilaclient import api_versions
from manilaclient import base
class ShareServer(base.Resource):

    def __repr__(self):
        return '<ShareServer: %s>' % self.id

    def __getattr__(self, attr):
        if attr == 'share_network':
            attr = 'share_network_name'
        return super(ShareServer, self).__getattr__(attr)

    def delete(self):
        """Delete this share server."""
        self.manager.delete(self)

    def unmanage(self, force=False):
        """Unmanage this share server."""
        self.manager.unmanage(self, force)

    def reset_state(self, state):
        """Update the share server with the provided state."""
        self.manager.reset_state(self, state)

    def migration_check(self, host, writable, nondisruptive, preserve_snapshots, new_share_network_id=None):
        """Check if the new host is suitable for migration."""
        return self.manager.migration_check(self, host, writable, nondisruptive, preserve_snapshots, new_share_network_id=new_share_network_id)

    def migration_start(self, host, writable, nondisruptive, preserve_snapshots, new_share_network_id=None):
        """Migrate the share server to a new host."""
        self.manager.migration_start(self, host, writable, nondisruptive, preserve_snapshots, new_share_network_id=new_share_network_id)

    def migration_complete(self):
        """Complete migration of a share server."""
        return self.manager.migration_complete(self)

    def migration_cancel(self):
        """Attempts to cancel migration of a share server."""
        self.manager.migration_cancel(self)

    def migration_get_progress(self):
        """Obtain progress of migration of a share server."""
        return self.manager.migration_get_progress(self)

    def reset_task_state(self, task_state):
        """Reset the task state of a given share server."""
        self.manager.reset_task_state(self, task_state)