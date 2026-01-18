import pathlib
from base64 import b64encode
from io import BytesIO
import numpy as np
import pytest
from panel.pane import Audio, Video
from panel.pane.media import TensorLike, _is_1dim_int_or_float_tensor
def test_video_url(document, comm):
    url = 'https://file-examples-com.github.io/uploads/2017/04/file_example_MP4_640_3MG.mp4'
    video = Video(url)
    model = video.get_root(document, comm=comm)
    assert model.value == url