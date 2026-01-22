from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum
import os
import uuid
import webbrowser
import cirq_web
def _get_bundle_script(self):
    """Returns the bundle script of a widget"""
    bundle_filename = self.get_widget_bundle_name()
    return _to_script_tag(bundle_filename)