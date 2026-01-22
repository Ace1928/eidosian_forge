import sys
import atexit
import pyglet
def _delete_audio_driver():
    global _audio_driver
    from .. import Source
    for p in Source._players:
        p.delete()
    del Source._players
    _audio_driver.delete()
    _audio_driver = None