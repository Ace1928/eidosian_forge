from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
@staticmethod
def _Train(arg=None, **kwargs):
    """Train Sentencepiece model. Accept both kwargs and legacy string arg."""
    if arg is not None and type(arg) is str:
        return SentencePieceTrainer._TrainFromString(arg)

    def _encode(value):
        """Encode value to CSV.."""
        if type(value) is list:
            if sys.version_info[0] == 3:
                f = StringIO()
            else:
                f = BytesIO()
            writer = csv.writer(f, lineterminator='')
            writer.writerow([str(v) for v in value])
            return f.getvalue()
        else:
            return str(value)
    sentence_iterator = None
    model_writer = None
    new_kwargs = {}
    for key, value in kwargs.items():
        if key in ['sentence_iterator', 'sentence_reader']:
            sentence_iterator = value
        elif key in ['model_writer']:
            model_writer = value
        else:
            new_kwargs[key] = _encode(value)
    if model_writer:
        if sentence_iterator:
            model_proto = SentencePieceTrainer._TrainFromMap4(new_kwargs, sentence_iterator)
        else:
            model_proto = SentencePieceTrainer._TrainFromMap3(new_kwargs)
        model_writer.write(model_proto)
    elif sentence_iterator:
        return SentencePieceTrainer._TrainFromMap2(new_kwargs, sentence_iterator)
    else:
        return SentencePieceTrainer._TrainFromMap(new_kwargs)
    return None