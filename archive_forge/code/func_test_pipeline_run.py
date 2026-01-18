import os.path as op
from pyxnat import Interface
from pyxnat.core.pipelines import PipelineNotFoundError
def test_pipeline_run():
    exp_id = 'BBRCDEV_E03094'
    try:
        wrong_pipe = p.pipelines.pipeline('INVALID_PIPELINE')
        wrong_pipe.run(exp_id)
    except PipelineNotFoundError as pe:
        print(pe)
    try:
        pipe = p.pipelines.pipeline('DicomToNifti')
        pipe.run('INVALID_EXP')
    except ValueError as ve:
        print(ve)
    pipe = p.pipelines.pipeline('DicomToNifti')
    pipe.run(exp_id)