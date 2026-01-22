from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import utils
class Consistencygroup(base.Resource):
    """A Consistencygroup of volumes."""

    def __repr__(self):
        return '<Consistencygroup: %s>' % self.id

    def delete(self, force='False'):
        """Delete this consistency group."""
        return self.manager.delete(self, force)

    def update(self, **kwargs):
        """Update the name or description for this consistency group."""
        return self.manager.update(self, **kwargs)