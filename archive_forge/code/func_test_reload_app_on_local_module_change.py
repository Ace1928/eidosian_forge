import os
import pathlib
import time
import pytest
from panel.io.state import state
from panel.tests.util import serve_component
@pytest.mark.flaky(reruns=3, reason='Writing files can sometimes be unpredictable')
def test_reload_app_on_local_module_change(page, autoreload, py_files):
    py_file, module = py_files
    import_name = pathlib.Path(module.name).stem
    module.write("var = 'foo';")
    module.close()
    py_file.write(f'import panel as pn; from {import_name} import var; print(var); pn.panel(var).servable();')
    py_file.close()
    path = pathlib.Path(py_file.name)
    autoreload(path)
    serve_component(page, path, warm=True)
    expect(page.locator('.markdown')).to_have_text('foo')
    time.sleep(0.1)
    with open(module.name, 'w') as f:
        f.write("var = 'bar';")
    pathlib.Path(module.name).touch()
    time.sleep(0.1)
    expect(page.locator('.markdown')).to_have_text('bar')