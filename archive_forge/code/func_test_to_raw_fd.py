import errno
import os
import socket
import pytest
from jeepney import FileDescriptor, NoFDError
def test_to_raw_fd(tmp_path):
    fd = os.open(tmp_path / 'a', os.O_CREAT)
    wfd = FileDescriptor(fd)
    assert wfd.fileno() == fd
    assert wfd.to_raw_fd() == fd
    try:
        assert 'converted' in repr(wfd)
        with pytest.raises(NoFDError):
            wfd.fileno()
    finally:
        os.close(fd)