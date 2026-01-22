from troveclient import base
class HwInfo(base.Resource):

    def __repr__(self):
        return '<HwInfo: %s>' % self.version