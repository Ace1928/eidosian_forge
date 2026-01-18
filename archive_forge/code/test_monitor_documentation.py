import zmq
import zmq.asyncio
from zmq.tests import require_zmq_4
from zmq.utils.monitor import recv_monitor_message
import pytest
Test connected monitoring socket.