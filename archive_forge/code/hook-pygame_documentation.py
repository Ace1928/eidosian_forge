import os
import platform
from pygame import __file__ as pygame_main_file

binaries hook for pygame seems to be required for pygame 2.0 Windows.
Otherwise some essential DLLs will not be transferred to the exe.

And also put hooks for datas, resources that pygame uses, to work
correctly with pyinstaller
