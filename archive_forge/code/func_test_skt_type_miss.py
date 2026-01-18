import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_skt_type_miss():

    @context()
    @socket('myskt')
    def f(ctx, myskt):
        pass
    with raises(TypeError):
        f()