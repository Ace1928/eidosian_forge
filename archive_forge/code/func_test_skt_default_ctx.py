import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
def test_skt_default_ctx():

    @socket(zmq.PUB)
    def test(skt):
        assert isinstance(skt, zmq.Socket), skt
        assert skt.context is zmq.Context.instance()
        assert skt.type == zmq.PUB
    test()