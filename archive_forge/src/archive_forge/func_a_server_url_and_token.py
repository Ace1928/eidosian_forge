import json
import os
import socket
import subprocess
import sys
import time
import uuid
from typing import Generator, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import pytest
from .conftest import KNOWN_SERVERS, extra_node_roots
@pytest.fixture(scope='module')
def a_server_url_and_token(tmp_path_factory: pytest.TempPathFactory) -> Generator[Tuple[str, str], None, None]:
    """Start a temporary, isolated jupyter server."""
    token = str(uuid.uuid4())
    port = get_unused_port()
    root_dir = tmp_path_factory.mktemp('root_dir')
    home = tmp_path_factory.mktemp('home')
    server_conf = home / 'etc/jupyter/jupyter_config.json'
    server_conf.parent.mkdir(parents=True)
    extensions = {'jupyter_lsp': True, 'jupyterlab': False, 'nbclassic': False}
    app = {'jpserver_extensions': extensions, 'token': token}
    lsm = {**extra_node_roots()}
    config_data = {'ServerApp': app, 'IdentityProvider': {'token': token}, 'LanguageServerManager': lsm}
    server_conf.write_text(json.dumps(config_data), encoding='utf-8')
    args = [*SUBPROCESS_PREFIX, 'jupyter_server', f'--port={port}', '--no-browser']
    print('server args', args)
    env = dict(os.environ)
    env.update(HOME=str(home), USERPROFILE=str(home), JUPYTER_CONFIG_DIR=str(server_conf.parent))
    proc = subprocess.Popen(args, cwd=str(root_dir), env=env, stdin=subprocess.PIPE)
    url = f'http://{LOCALHOST}:{port}'
    retries = 20
    ok = False
    while not ok and retries:
        try:
            ok = urlopen(f'{url}/favicon.ico')
        except URLError:
            print(f'[{retries} / 20] ...', flush=True)
            retries -= 1
            time.sleep(1)
    if not ok:
        raise RuntimeError('the server did not start')
    yield (url, token)
    try:
        print('shutting down with API...')
        urlopen(f'{url}/api/shutdown?token={token}', data=[])
    except URLError:
        print('shutting down the hard way...')
        proc.terminate()
        proc.communicate(b'y\n')
        proc.wait()
        proc.kill()
    proc.wait()
    assert proc.returncode is not None, 'jupyter-server probably still running'