import sys
from traitlets.config.configurable import Configurable
from traitlets import List
from .display_functions import publish_display_data
import typing as t
class CapturingDisplayPublisher(DisplayPublisher):
    """A DisplayPublisher that stores"""
    outputs: List = List()

    def publish(self, data, metadata=None, source=None, *, transient=None, update=False):
        self.outputs.append({'data': data, 'metadata': metadata, 'transient': transient, 'update': update})

    def clear_output(self, wait=False):
        super(CapturingDisplayPublisher, self).clear_output(wait)
        self.outputs.clear()