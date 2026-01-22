from __future__ import annotations
import csv
import datetime
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any
import filelock
import huggingface_hub
from gradio_client import utils as client_utils
from gradio_client.documentation import document
import gradio as gr
from gradio import utils
class FlagMethod:
    """
    Helper class that contains the flagging options and calls the flagging method. Also
    provides visual feedback to the user when flag is clicked.
    """

    def __init__(self, flagging_callback: FlaggingCallback, label: str, value: str, visual_feedback: bool=True):
        self.flagging_callback = flagging_callback
        self.label = label
        self.value = value
        self.__name__ = 'Flag'
        self.visual_feedback = visual_feedback

    def __call__(self, request: gr.Request, *flag_data):
        try:
            self.flagging_callback.flag(list(flag_data), flag_option=self.value, username=request.username)
        except Exception as e:
            print(f'Error while flagging: {e}')
            if self.visual_feedback:
                return 'Error!'
        if not self.visual_feedback:
            return
        time.sleep(0.8)
        return self.reset()

    def reset(self):
        return gr.Button(value=self.label, interactive=True)