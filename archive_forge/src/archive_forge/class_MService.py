from magnumclient.common import base
from magnumclient.common import utils
class MService(base.Resource):

    def __repr__(self):
        return '<Service %s>' % self._info