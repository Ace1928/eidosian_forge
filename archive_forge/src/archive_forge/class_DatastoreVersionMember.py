from troveclient import base
class DatastoreVersionMember(base.Resource):

    def __repr__(self):
        return '<DatastoreVersionMember: %s>' % self.id