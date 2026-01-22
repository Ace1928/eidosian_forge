import atexit
import pyglet
def _delete_manager():
    """Deletes existing manager. If audio device manager is stored anywhere.
    Required to remove handlers before exit, as it can cause problems with the event system's weakrefs."""
    global _audio_device_manager
    _audio_device_manager = None