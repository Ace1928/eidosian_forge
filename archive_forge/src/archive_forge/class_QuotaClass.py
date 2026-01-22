from zunclient.common import base
class QuotaClass(base.Resource):

    def __repr__(self):
        return '<QuotaClass %s>' % self._info