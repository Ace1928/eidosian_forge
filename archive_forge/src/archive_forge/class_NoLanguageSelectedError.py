import sys
from kivy.core import core_select_lib
class NoLanguageSelectedError(Exception):
    """
    Exception to be raised when a language-using method is called but no
    language was selected prior to the call.
    """
    pass