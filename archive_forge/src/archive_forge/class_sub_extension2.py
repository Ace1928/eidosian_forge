import pytest
import pyviz_comms
from pyviz_comms import extension
class sub_extension2(extension):

    def __call__(self, *args, **params):
        pass