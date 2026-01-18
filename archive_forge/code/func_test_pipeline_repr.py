import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
def test_pipeline_repr():
    pipeline = Pipeline()
    pipeline.add_stage('Stage 1', Stage1)
    pipeline.add_stage('Stage 2', Stage2)
    assert repr(pipeline) == 'Pipeline:\n    [0] Stage 1: Stage1()\n    [1] Stage 2: Stage2()'