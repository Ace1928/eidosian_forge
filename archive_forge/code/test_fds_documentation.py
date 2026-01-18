import errno
import os
import socket
import pytest
from jeepney import FileDescriptor, NoFDError
Check that the given number is not open as a file descriptor