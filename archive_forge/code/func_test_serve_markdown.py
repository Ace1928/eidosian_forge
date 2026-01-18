import os
import tempfile
import pytest
import requests
from panel.tests.util import (
import panel as pn
@linux_only
def test_serve_markdown():
    md = tempfile.NamedTemporaryFile(mode='w', suffix='.md')
    write_file(md_app, md.file)
    with run_panel_serve(['--port', '0', md.name]) as p:
        port = wait_for_port(p.stdout)
        r = requests.get(f'http://localhost:{port}/')
        assert r.status_code == 200
        assert '<title>My app</title>' in r.content.decode('utf-8')