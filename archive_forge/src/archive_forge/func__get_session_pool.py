import logging
from string import Template
from typing import Any, Dict, Optional
def _get_session_pool(self) -> Any:
    assert all([self.username, self.password, self.address, self.port, self.space]), 'Please provide all of the following parameters: username, password, address, port, space'
    from nebula3.Config import SessionPoolConfig
    from nebula3.Exception import AuthFailedException, InValidHostname
    from nebula3.gclient.net.SessionPool import SessionPool
    config = SessionPoolConfig()
    config.max_size = self.session_pool_size
    try:
        session_pool = SessionPool(self.username, self.password, self.space, [(self.address, self.port)])
    except InValidHostname:
        raise ValueError('Could not connect to NebulaGraph database. Please ensure that the address and port are correct')
    try:
        session_pool.init(config)
    except AuthFailedException:
        raise ValueError('Could not connect to NebulaGraph database. Please ensure that the username and password are correct')
    except RuntimeError as e:
        raise ValueError(f'Error initializing session pool. Error: {e}')
    return session_pool