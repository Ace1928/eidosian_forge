from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Any, Callable, Literal
import httpx
import numpy as np
from gradio_client import file
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import processing_utils, utils
from gradio.components.base import Component, StreamingInput, StreamingOutput
from gradio.data_classes import FileData
from gradio.events import Events
from gradio.exceptions import Error
@dataclasses.dataclass
class WaveformOptions:
    """
    A dataclass for specifying options for the waveform display in the Audio component. An instance of this class can be passed into the `waveform_options` parameter of `gr.Audio`.
    Parameters:
        waveform_color: The color (as a hex string or valid CSS color) of the full waveform representing the amplitude of the audio. Defaults to a light gray color.
        waveform_progress_color: The color (as a hex string or valid CSS color) that the waveform fills with to as the audio plays. Defaults to the accent color.
        trim_region_color: The color (as a hex string or valid CSS color) of the trim region. Defaults to the accent color.
        show_recording_waveform: Whether to show the waveform when recording audio. Defaults to True.
        show_controls: Whether to show the standard HTML audio player below the waveform when recording audio or playing recorded audio. Defaults to False.
        skip_length: The percentage (between 0 and 100) of the audio to skip when clicking on the skip forward / skip backward buttons. Defaults to 5.
        sample_rate: The output sample rate (in Hz) of the audio after editing. Defaults to 44100.
    """
    waveform_color: str | None = None
    waveform_progress_color: str | None = None
    trim_region_color: str | None = None
    show_recording_waveform: bool = True
    show_controls: bool = False
    skip_length: int | float = 5
    sample_rate: int = 44100