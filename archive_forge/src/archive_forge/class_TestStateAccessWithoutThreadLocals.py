import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
class TestStateAccessWithoutThreadLocals(PecanTestCase):

    def setUp(self):
        super(TestStateAccessWithoutThreadLocals, self).setUp()
        self.args = None

        class RootController(object):

            @expose()
            def index(self, req, resp):
                return 'Hello, World!'

            @expose()
            def greet(self, req, resp, name):
                return 'Hello, %s!' % name

            @expose()
            def greetmore(self, req, resp, *args):
                return 'Hello, %s!' % args[0]

            @expose()
            def kwargs(self, req, resp, **kw):
                return 'Hello, %s!' % kw['name']

            @expose()
            def mixed(self, req, resp, first, second, *args):
                return 'Mixed'

        class SimpleHook(PecanHook):

            def before(inself, state):
                self.args = (state.controller, state.arguments)
        self.root = RootController()
        self.app = TestApp(make_app(self.root, hooks=[SimpleHook()], use_context_locals=False))

    def test_no_args(self):
        self.app.get('/')
        assert self.args[0] == self.root.index
        assert isinstance(self.args[1], inspect.Arguments)
        assert len(self.args[1].args) == 2
        assert isinstance(self.args[1].args[0], Request)
        assert isinstance(self.args[1].args[1], Response)
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {}

    def test_single_arg(self):
        self.app.get('/greet/joe')
        assert self.args[0] == self.root.greet
        assert isinstance(self.args[1], inspect.Arguments)
        assert len(self.args[1].args) == 3
        assert isinstance(self.args[1].args[0], Request)
        assert isinstance(self.args[1].args[1], Response)
        assert self.args[1].args[2] == 'joe'
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {}

    def test_single_vararg(self):
        self.app.get('/greetmore/joe')
        assert self.args[0] == self.root.greetmore
        assert isinstance(self.args[1], inspect.Arguments)
        assert len(self.args[1].args) == 2
        assert isinstance(self.args[1].args[0], Request)
        assert isinstance(self.args[1].args[1], Response)
        assert self.args[1].varargs == ['joe']
        assert kwargs(self.args[1]) == {}

    def test_single_kw(self):
        self.app.get('/kwargs/?name=joe')
        assert self.args[0] == self.root.kwargs
        assert isinstance(self.args[1], inspect.Arguments)
        assert len(self.args[1].args) == 2
        assert isinstance(self.args[1].args[0], Request)
        assert isinstance(self.args[1].args[1], Response)
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {'name': 'joe'}

    def test_single_kw_post(self):
        self.app.post('/kwargs/', params={'name': 'joe'})
        assert self.args[0] == self.root.kwargs
        assert isinstance(self.args[1], inspect.Arguments)
        assert len(self.args[1].args) == 2
        assert isinstance(self.args[1].args[0], Request)
        assert isinstance(self.args[1].args[1], Response)
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {'name': 'joe'}

    def test_mixed_args(self):
        self.app.get('/mixed/foo/bar/spam/eggs')
        assert self.args[0] == self.root.mixed
        assert isinstance(self.args[1], inspect.Arguments)
        assert len(self.args[1].args) == 4
        assert isinstance(self.args[1].args[0], Request)
        assert isinstance(self.args[1].args[1], Response)
        assert self.args[1].args[2:] == ['foo', 'bar']
        assert self.args[1].varargs == ['spam', 'eggs']