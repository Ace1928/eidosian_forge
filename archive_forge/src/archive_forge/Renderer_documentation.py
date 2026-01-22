from contextlib import contextmanager
from ipywidgets import widget_serialization
from traitlets import (
from ..traits import *
from .._base.renderable import RenderableWidget
from ..scenes.Scene_autogen import Scene
from ..cameras.Camera_autogen import Camera
from ..controls.Controls_autogen import Controls
from_json = widget_serialization['from_json']
from inspect import Signature, Parameter
Renderer
    