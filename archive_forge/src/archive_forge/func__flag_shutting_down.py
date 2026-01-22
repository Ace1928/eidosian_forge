import atexit
@atexit.register
def _flag_shutting_down():
    python_is_shutting_down.isalive.clear()