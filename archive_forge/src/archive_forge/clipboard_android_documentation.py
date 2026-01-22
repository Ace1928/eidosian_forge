from kivy import Logger
from kivy.core.clipboard import ClipboardBase
from jnius import autoclass, cast
from android.runnable import run_on_ui_thread
from android import python_act

Clipboard Android
=================

Android implementation of Clipboard provider, using Pyjnius.
