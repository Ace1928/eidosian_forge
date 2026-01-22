from taskflow.listeners import base
@staticmethod
def _format_capture(kind, state, details):
    """Tweak what is saved according to your desire(s)."""
    return (kind, state, details)