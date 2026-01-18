import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
def test_pipeline_previous_follows_initial_path():
    pipeline = Pipeline()
    pipeline.add_stage('Stage 1', Stage1)
    pipeline.add_stage('Stage 2', Stage2)
    pipeline.add_stage('Stage 2b', Stage2b)
    pipeline.add_stage('Stage 3', DummyStage)
    pipeline.define_graph({'Stage 1': ('Stage 2', 'Stage 2b'), 'Stage 2': 'Stage 3', 'Stage 2b': 'Stage 3'})
    assert pipeline._route == ['Stage 1']
    pipeline.next_selector.value = 'Stage 2b'
    pipeline._next()
    assert pipeline._route == ['Stage 1', 'Stage 2b']
    pipeline._next()
    assert pipeline._route == ['Stage 1', 'Stage 2b', 'Stage 3']
    pipeline._previous()
    assert pipeline._stage == 'Stage 2b'
    assert pipeline._route == ['Stage 1', 'Stage 2b']