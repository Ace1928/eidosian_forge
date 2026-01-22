from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import utils
class Cgsnapshot(base.Resource):
    """A cgsnapshot is snapshot of a consistency group."""

    def __repr__(self):
        return '<cgsnapshot: %s>' % self.id

    def delete(self):
        """Delete this cgsnapshot."""
        return self.manager.delete(self)

    def update(self, **kwargs):
        """Update the name or description for this cgsnapshot."""
        return self.manager.update(self, **kwargs)