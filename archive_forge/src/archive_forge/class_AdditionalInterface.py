from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
class AdditionalInterface(typing.TypedDict):
    refs: list[str]
    source: str