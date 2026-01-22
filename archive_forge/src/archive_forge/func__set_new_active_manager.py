import atexit
from collections import OrderedDict
@classmethod
def _set_new_active_manager(cls, manager):
    """Adopt *manager* into pyplot and make it the active manager."""
    if not hasattr(manager, '_cidgcf'):
        manager._cidgcf = manager.canvas.mpl_connect('button_press_event', lambda event: cls.set_active(manager))
    fig = manager.canvas.figure
    fig.number = manager.num
    label = fig.get_label()
    if label:
        manager.set_window_title(label)
    cls.set_active(manager)