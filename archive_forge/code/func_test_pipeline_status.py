import os.path as op
from pyxnat import Interface
from pyxnat.core.pipelines import PipelineNotFoundError
def test_pipeline_status():
    exp_id = 'BBRCDEV_E03094'
    try:
        wrong_pipe = p.pipelines.pipeline('INVALID_PIPELINE')
        wrong_pipe.status(exp_id)
    except PipelineNotFoundError as pe:
        print(pe)
    try:
        pipe = p.pipelines.pipeline('DicomToNifti')
        pipe.status('INVALID_EXP')
    except ValueError as ve:
        print(ve)
    pipe = p.pipelines.pipeline('DicomToNifti')
    status = pipe.status(exp_id)
    assert isinstance(status, dict)
    assert status['status'] != 'Failed'