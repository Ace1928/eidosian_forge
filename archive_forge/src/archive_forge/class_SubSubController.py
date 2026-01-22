from json import dumps
from webtest import TestApp
from pecan import Pecan, expose, abort
from pecan.tests import PecanTestCase
class SubSubController(object):

    @expose(generic=True)
    def index(self):
        return 'GET'

    @index.when(method='DELETE', template='json')
    def do_delete(self, name, *args):
        return dict(result=name, args=', '.join(args))