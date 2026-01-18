from io import BytesIO
from typing import List, Union
import requests
from ..utils import add_end_docstrings, is_decord_available, is_torch_available, logging, requires_backends
from .base import Pipeline, build_pipeline_init_args

        Assign labels to the video(s) passed as inputs.

        Args:
            videos (`str`, `List[str]`):
                The pipeline handles three types of videos:

                - A string containing a http link pointing to a video
                - A string containing a local path to a video

                The pipeline accepts either a single video or a batch of videos, which must then be passed as a string.
                Videos in a batch must all be in the same format: all as http links or all as local paths.
            top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.
            num_frames (`int`, *optional*, defaults to `self.model.config.num_frames`):
                The number of frames sampled from the video to run the classification on. If not provided, will default
                to the number of frames specified in the model configuration.
            frame_sampling_rate (`int`, *optional*, defaults to 1):
                The sampling rate used to select frames from the video. If not provided, will default to 1, i.e. every
                frame will be used.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single video, will return a
            dictionary, if the input is a list of several videos, will return a list of dictionaries corresponding to
            the videos.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        