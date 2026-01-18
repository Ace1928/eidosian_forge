import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
def test_pipeline_network_diagram_states():
    pipeline = Pipeline(ready_parameter='ready', auto_advance=True)
    pipeline.add_stage('Stage 1', Stage1)
    pipeline.add_stage('Stage 2', Stage2)
    pipeline.add_stage('Stage 2b', Stage2b)
    pipeline.define_graph({'Stage 1': ('Stage 2', 'Stage 2b')})
    [s1, s2, s2b] = pipeline.network.object.get(0).nodes['State']
    assert s1 == 'active'
    assert s2 == 'inactive'
    assert s2b == 'next'
    pipeline._next()
    [s1, s2, s2b] = pipeline.network.object.get(0).nodes['State']
    assert s1 == 'inactive'
    assert s2 == 'inactive'
    assert s2b == 'active'
    pipeline._previous()
    [s1, s2, s2b] = pipeline.network.object.get(0).nodes['State']
    assert s1 == 'active'
    assert s2 == 'inactive'
    assert s2b == 'next'