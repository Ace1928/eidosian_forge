from manilaclient import api_versions
from manilaclient import base
class ShareTypeAccess(base.Resource):

    def __repr__(self):
        return '<ShareTypeAccess: %s>' % self.id