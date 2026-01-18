import os.path
from .exceptions import NoSuchWidgetError, WidgetPluginError
 Load the plugin from the given file.  Return True if the plugin was
        loaded, or False if it wanted to be ignored.  Raise an exception if
        there was an error.
        