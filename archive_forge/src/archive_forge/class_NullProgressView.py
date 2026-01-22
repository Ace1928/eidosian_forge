import warnings
class NullProgressView:
    """Soak up and ignore progress information."""

    def clear(self):
        pass

    def show_progress(self, task):
        pass

    def show_transport_activity(self, transport, direction, byte_count):
        pass

    def log_transport_activity(self, display=False):
        pass