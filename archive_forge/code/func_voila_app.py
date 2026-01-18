import os
import voila.app
import pytest
@pytest.fixture
def voila_app(voila_args, voila_config):
    voila_app = VoilaTest.instance()
    voila_app.initialize(voila_args + ['--no-browser', '--template=gridstack'])
    voila_config(voila_app)
    voila_app.start()
    yield voila_app
    voila_app.stop()
    voila_app.clear_instance()