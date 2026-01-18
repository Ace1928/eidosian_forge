import os.path as op
from pyxnat import Interface
from pyxnat.core.pipelines import PipelineNotFoundError
def test_pipeline_info():
    info_keys = {'appliesTo', 'authors', 'description', 'inputParameters', 'path', 'resourceRequirements', 'steps', 'version'}
    pipe = p.pipelines.pipeline('DicomToNifti')
    assert pipe.exists()
    pipe_info = pipe.info()
    assert int(pipe_info['version']) >= 20190308
    assert set(pipe_info.keys()) == info_keys