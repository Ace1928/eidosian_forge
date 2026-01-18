import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
def test_pipeline_ready_respected():
    pipeline = Pipeline(ready_parameter='ready')
    pipeline.add_stage('Stage 1', Stage1)
    pipeline.add_stage('Stage 2', Stage2)
    assert pipeline.next_button.disabled
    pipeline._state.ready = True
    assert not pipeline.next_button.disabled