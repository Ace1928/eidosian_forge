import os
import sys
import time
from pytest import mark
import zmq
from zmq.tests import GreenTest, PollZMQTestCase, have_gevent
make sure select timeout has the right units (seconds).