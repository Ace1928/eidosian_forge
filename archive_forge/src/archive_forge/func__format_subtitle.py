from __future__ import annotations
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, Optional
from gradio_client import file
from gradio_client import utils as client_utils
from gradio_client.documentation import document
import gradio as gr
from gradio import processing_utils, utils, wasm_utils
from gradio.components.base import Component
from gradio.data_classes import FileData, GradioModel
from gradio.events import Events
def _format_subtitle(self, subtitle: str | Path | None) -> FileData | None:
    """
        Convert subtitle format to VTT and process the video to ensure it meets the HTML5 requirements.
        """

    def srt_to_vtt(srt_file_path, vtt_file_path):
        """Convert an SRT subtitle file to a VTT subtitle file"""
        with open(srt_file_path, encoding='utf-8') as srt_file, open(vtt_file_path, 'w', encoding='utf-8') as vtt_file:
            vtt_file.write('WEBVTT\n\n')
            for subtitle_block in srt_file.read().strip().split('\n\n'):
                subtitle_lines = subtitle_block.split('\n')
                subtitle_timing = subtitle_lines[1].replace(',', '.')
                subtitle_text = '\n'.join(subtitle_lines[2:])
                vtt_file.write(f'{subtitle_timing} --> {subtitle_timing}\n')
                vtt_file.write(f'{subtitle_text}\n\n')
    if subtitle is None:
        return None
    valid_extensions = ('.srt', '.vtt')
    if Path(subtitle).suffix not in valid_extensions:
        raise ValueError(f'Invalid value for parameter `subtitle`: {subtitle}. Please choose a file with one of these extensions: {valid_extensions}')
    if Path(subtitle).suffix == '.srt':
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.vtt', dir=self.GRADIO_CACHE)
        srt_to_vtt(subtitle, temp_file.name)
        subtitle = temp_file.name
    return FileData(path=str(subtitle))