import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
def test_pipeline_set_stage():
    pipeline = Pipeline()
    pipeline.add_stage('Stage 2', Stage2)
    pipeline.add_stage('Stage 2b', Stage2b)
    pipeline.add_stage('Stage 1', Stage1)
    pipeline.add_stage('Final', DummyStage())
    pipeline.define_graph({'Stage 1': ('Stage 2', 'Stage 2b'), 'Stage 2b': 'Final'})
    pipeline._set_stage([2])
    assert pipeline._stage == 'Stage 2'
    pipeline._set_stage([0])
    assert pipeline._stage == 'Stage 1'
    pipeline._set_stage([3])
    assert pipeline._stage == 'Final'