from ._base import *
class FileExtIO(Enum):
    TXT = ['txt', 'text']
    JSON = ['json']
    JSONLINES = ['jsonl', 'jsonlines', 'jlines']
    TORCH = ['bin', '.model']
    PICKLE = ['pkl', 'pickle', 'pb']
    NUMPY = ['numpy', 'npy']
    TFRECORDS = ['tfrecord', 'tfrecords', 'tfr']