import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
def test_pipeline_add_stage_validate_wrong_type():
    pipeline = Pipeline()
    with pytest.raises(ValueError):
        pipeline.add_stage('Stage 1', 1)