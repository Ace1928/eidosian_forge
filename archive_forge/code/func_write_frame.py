from fractions import Fraction
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
import av
import av.filter
import numpy as np
from numpy.lib.stride_tricks import as_strided
from ..core import Request
from ..core.request import URI_BYTES, InitializationError, IOMode
from ..core.v3_plugin_api import ImageProperties, PluginV3
def write_frame(self, frame: np.ndarray, *, pixel_format: str='rgb24') -> None:
    """Add a frame to the video stream.

        This function appends a new frame to the video. It assumes that the
        stream previously has been initialized. I.e., ``init_video_stream`` has
        to be called before calling this function for the write to succeed.

        Parameters
        ----------
        frame : np.ndarray
            The image to be appended/written to the video stream.
        pixel_format : str
            The colorspace (pixel format) of the incoming frame.

        Notes
        -----
        Frames may be held in a buffer, e.g., by the filter pipeline used during
        writing or by FFMPEG to batch them prior to encoding. Make sure to
        ``.close()`` the plugin or to use a context manager to ensure that all
        frames are written to the ImageResource.

        """
    pixel_format = av.VideoFormat(pixel_format)
    img_dtype = _format_to_dtype(pixel_format)
    width = frame.shape[2 if pixel_format.is_planar else 1]
    height = frame.shape[1 if pixel_format.is_planar else 0]
    av_frame = av.VideoFrame(width, height, pixel_format.name)
    if pixel_format.is_planar:
        for idx, plane in enumerate(av_frame.planes):
            plane_array = np.frombuffer(plane, dtype=img_dtype)
            plane_array = as_strided(plane_array, shape=(plane.height, plane.width), strides=(plane.line_size, img_dtype.itemsize))
            plane_array[...] = frame[idx]
    else:
        if pixel_format.name.startswith('bayer_'):
            n_channels = 1
        else:
            n_channels = len(pixel_format.components)
        plane = av_frame.planes[0]
        plane_shape = (plane.height, plane.width)
        plane_strides = (plane.line_size, n_channels * img_dtype.itemsize)
        if n_channels > 1:
            plane_shape += (n_channels,)
            plane_strides += (img_dtype.itemsize,)
        plane_array = as_strided(np.frombuffer(plane, dtype=img_dtype), shape=plane_shape, strides=plane_strides)
        plane_array[...] = frame
    stream = self._video_stream
    av_frame.time_base = stream.codec_context.time_base
    av_frame.pts = self.frames_written
    self.frames_written += 1
    if self._video_filter is not None:
        av_frame = self._video_filter.send(av_frame)
        if av_frame is None:
            return
    if stream.frames == 0:
        stream.width = av_frame.width
        stream.height = av_frame.height
    for packet in stream.encode(av_frame):
        self._container.mux(packet)