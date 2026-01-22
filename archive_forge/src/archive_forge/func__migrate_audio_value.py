import numpy as np
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.util import tensor_util
def _migrate_audio_value(value):
    audio_value = value.audio
    data = [[audio_value.encoded_audio_string, b'']]
    summary_metadata = audio_metadata.create_summary_metadata(display_name=value.metadata.display_name or value.tag, description=value.metadata.summary_description, encoding=audio_metadata.Encoding.Value('WAV'), converted_to_tensor=True)
    return make_summary(value.tag, summary_metadata, data)