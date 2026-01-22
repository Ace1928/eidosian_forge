from kivy.lib.gstplayer import GstPlayer, get_gst_version
from kivy.core.audio import Sound, SoundLoader
from kivy.logger import Logger
from kivy.compat import PY2
from kivy.clock import Clock
from os.path import realpath
def _on_gstplayer_message(mtype, message):
    if mtype == 'error':
        Logger.error('AudioGstplayer: {}'.format(message))
    elif mtype == 'warning':
        Logger.warning('AudioGstplayer: {}'.format(message))
    elif mtype == 'info':
        Logger.info('AudioGstplayer: {}'.format(message))