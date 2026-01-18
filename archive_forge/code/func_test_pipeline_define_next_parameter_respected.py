import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
def test_pipeline_define_next_parameter_respected():
    pipeline = Pipeline()
    pipeline.add_stage('Stage 2', Stage2)
    pipeline.add_stage('Stage 2b', Stage2b)
    pipeline.add_stage('Stage 1', Stage1(next='Stage 2b'), next_parameter='next')
    pipeline.define_graph({'Stage 1': ('Stage 2', 'Stage 2b')})
    assert pipeline.next_selector.value == 'Stage 2b'
    pipeline._state.next = 'Stage 2'
    assert pipeline.next_selector.value == 'Stage 2'