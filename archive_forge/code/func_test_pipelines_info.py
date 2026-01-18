import os.path as op
from pyxnat import Interface
from pyxnat.core.pipelines import PipelineNotFoundError
def test_pipelines_info():
    info = p.pipelines.info('DicomToNifti')
    assert len(info) == 1