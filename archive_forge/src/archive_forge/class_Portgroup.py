from ironicclient.common import base
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
class Portgroup(base.Resource):

    def __repr__(self):
        return '<Portgroup %s>' % self._info