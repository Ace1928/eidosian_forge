from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
class ShareBackup(base.Resource):

    def __repr__(self):
        return '<Share Backup: %s>' % self.id