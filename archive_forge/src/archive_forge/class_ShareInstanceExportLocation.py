from manilaclient import api_versions
from manilaclient import base
class ShareInstanceExportLocation(base.Resource):
    """Resource class for a share export location."""

    def __repr__(self):
        return '<ShareInstanceExportLocation: %s>' % self.id

    def __getitem__(self, key):
        return self._info[key]