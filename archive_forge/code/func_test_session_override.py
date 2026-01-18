import pytest
from panel import config, state
from panel.pane import HTML, panel
from panel.param import ParamFunction
from panel.tests.conftest import set_env_var
@pytest.mark.usefixtures('with_curdoc')
def test_session_override():
    config.sizing_mode = 'stretch_width'
    assert config.sizing_mode == 'stretch_width'
    assert state.curdoc in config._session_config
    assert config._session_config[state.curdoc] == {'sizing_mode': 'stretch_width'}
    state.curdoc = None
    assert config.sizing_mode is None