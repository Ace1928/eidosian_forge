from .. import progress, ui
class ProgressRecordingUIFactory(ui.UIFactory, progress.DummyProgress):
    """Captures progress updates made through it.

    This is overloaded as both the UIFactory and the progress model."""

    def __init__(self):
        super().__init__()
        self._calls = []
        self.depth = 0

    def nested_progress_bar(self):
        self.depth += 1
        return self

    def finished(self):
        self.depth -= 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finished()
        return False

    def update(self, message, count=None, total=None):
        if self.depth == 1:
            self._calls.append(('update', count, total, message))