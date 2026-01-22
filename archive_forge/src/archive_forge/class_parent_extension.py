import pytest
import pyviz_comms
from pyviz_comms import extension
class parent_extension(extension):

    def __call__(self, *args, **params):
        pass