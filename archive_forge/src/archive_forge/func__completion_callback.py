from jnius import autoclass, java_method, PythonJavaClass
from android import api_version
from kivy.core.audio import Sound, SoundLoader
def _completion_callback(self):
    super(SoundAndroidPlayer, self).stop()