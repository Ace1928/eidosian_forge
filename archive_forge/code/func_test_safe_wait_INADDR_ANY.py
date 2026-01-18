import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
def test_safe_wait_INADDR_ANY():
    """
    Wait on INADDR_ANY should not raise IOError

    In cases where the loopback interface does not exist, CherryPy cannot
    effectively determine if a port binding to INADDR_ANY was effected.
    In this situation, CherryPy should assume that it failed to detect
    the binding (not that the binding failed) and only warn that it could
    not verify it.
    """
    free_port = portend.find_available_local_port()
    servers = cherrypy.process.servers
    inaddr_any = '0.0.0.0'
    with pytest.warns(UserWarning, match='Unable to verify that the server is bound on ') as warnings:
        with servers._safe_wait(inaddr_any, free_port):
            portend.occupied(inaddr_any, free_port, timeout=1)
    assert len(warnings) == 1
    with pytest.raises(IOError):
        with servers._safe_wait('127.0.0.1', free_port):
            portend.occupied('127.0.0.1', free_port, timeout=1)