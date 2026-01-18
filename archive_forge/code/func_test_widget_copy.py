import inspect
import pytest
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display
from IPython.utils.capture import capture_output
from .. import widget
from ..widget import Widget
from ..widget_button import Button
import copy
def test_widget_copy():
    button = Button()
    with pytest.raises(NotImplementedError):
        copy.copy(button)
    with pytest.raises(NotImplementedError):
        copy.deepcopy(button)