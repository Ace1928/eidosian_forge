from ipywidgets import register, DOMWidget, Widget
from traitlets import Unicode, Union, CInt
from ..traits import IEEEFloat, ieee_float_serializers
from .._package import npm_pkg_name
from .._version import EXTENSION_SPEC_VERSION
from .._base.Three import ThreeWidget
from .AnimationAction_autogen import AnimationAction as AnimationActionBase
AnimationAction is a three widget that also has its own view.

    The view offers animation action controls.
    