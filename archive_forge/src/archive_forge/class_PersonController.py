from pecan import expose, response, abort
class PersonController(object):

    def __init__(self, person_id):
        self.person_id = person_id

    @expose(generic=True)
    def index(self):
        return people.get(self.person_id) or abort(404)

    @index.when(method='PUT')
    def put(self):
        response.status = 204

    @index.when(method='DELETE')
    def delete(self):
        response.status = 204