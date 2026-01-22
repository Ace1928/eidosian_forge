from zunclient.common import base
from zunclient.common import utils
from zunclient import exceptions
class Capsule(base.Resource):

    def __repr__(self):
        return '<Capsule %s>' % self._info