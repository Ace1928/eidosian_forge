import time
import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, have_gevent
class TestPubSubGreen(GreenTest, TestPubSub):
    pass