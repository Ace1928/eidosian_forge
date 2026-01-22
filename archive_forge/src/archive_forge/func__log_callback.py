import ffpyplayer
from ffpyplayer.pic import ImageLoader as ffImageLoader, SWScale
from ffpyplayer.tools import set_log_callback, get_log_callback
from kivy.logger import Logger
from kivy.core.image import ImageLoaderBase, ImageData, ImageLoader
def _log_callback(message, level):
    message = message.strip()
    if message:
        logger_func[level]('ffpyplayer: {}'.format(message))