import errno
import re
import socket
import sys
import pytest
import trio
from trio.testing._fake_net import FakeNet
Test all recv methods for codecov