from unittest import mock
from os_brick import exception
from os_brick.initiator import linuxrbd
from os_brick.tests import base
from os_brick import utils
Test when image is closed but wrapper isn't