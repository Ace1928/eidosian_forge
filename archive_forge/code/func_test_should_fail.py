import os
import pytest
from nipype.interfaces import utility
import nipype.pipeline.engine as pe
def test_should_fail(tmpdir):
    with pytest.raises(pe.nodes.NodeExecutionError):
        should_fail(tmpdir)