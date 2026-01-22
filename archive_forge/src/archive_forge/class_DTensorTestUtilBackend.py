import enum
import os
class DTensorTestUtilBackend(enum.Enum):
    """DTensor backend the test is being run on."""
    UNSPECIFIED = 'unspecified'
    CPU = 'cpu'
    GPU = 'gpu'
    GPU_2DEVS_BACKEND = '2gpus'
    TPU = 'tpu'
    TPU_STREAM_EXECUTOR = 'tpu_se'
    TPU_V3_DONUT_BACKEND = 'tpu_v3_2x2'
    TPU_V4_DONUT_BACKEND = 'tpu_v4_2x2'
    PATHWAYS = 'pw'
    PATHWAYS_V3_DONUT_BACKEND = 'pw_v3_2x2'