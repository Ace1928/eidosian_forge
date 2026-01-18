import abc
import base64
import contextlib
from io import BytesIO, TextIOWrapper
import itertools
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import uuid
import warnings
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib._animation_data import (
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
def to_html5_video(self, embed_limit=None):
    """
        Convert the animation to an HTML5 ``<video>`` tag.

        This saves the animation as an h264 video, encoded in base64
        directly into the HTML5 video tag. This respects :rc:`animation.writer`
        and :rc:`animation.bitrate`. This also makes use of the
        *interval* to control the speed, and uses the *repeat*
        parameter to decide whether to loop.

        Parameters
        ----------
        embed_limit : float, optional
            Limit, in MB, of the returned animation. No animation is created
            if the limit is exceeded.
            Defaults to :rc:`animation.embed_limit` = 20.0.

        Returns
        -------
        str
            An HTML5 video tag with the animation embedded as base64 encoded
            h264 video.
            If the *embed_limit* is exceeded, this returns the string
            "Video too large to embed."
        """
    VIDEO_TAG = '<video {size} {options}>\n  <source type="video/mp4" src="data:video/mp4;base64,{video}">\n  Your browser does not support the video tag.\n</video>'
    if not hasattr(self, '_base64_video'):
        embed_limit = mpl._val_or_rc(embed_limit, 'animation.embed_limit')
        embed_limit *= 1024 * 1024
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, 'temp.m4v')
            Writer = writers[mpl.rcParams['animation.writer']]
            writer = Writer(codec='h264', bitrate=mpl.rcParams['animation.bitrate'], fps=1000.0 / self._interval)
            self.save(str(path), writer=writer)
            vid64 = base64.encodebytes(path.read_bytes())
        vid_len = len(vid64)
        if vid_len >= embed_limit:
            _log.warning("Animation movie is %s bytes, exceeding the limit of %s. If you're sure you want a large animation embedded, set the animation.embed_limit rc parameter to a larger value (in MB).", vid_len, embed_limit)
        else:
            self._base64_video = vid64.decode('ascii')
            self._video_size = 'width="{}" height="{}"'.format(*writer.frame_size)
    if hasattr(self, '_base64_video'):
        options = ['controls', 'autoplay']
        if getattr(self, '_repeat', False):
            options.append('loop')
        return VIDEO_TAG.format(video=self._base64_video, size=self._video_size, options=' '.join(options))
    else:
        return 'Video too large to embed.'