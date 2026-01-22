import logging
import sys
import warnings
from typing import Optional
import wandb
class ProgressWidget:
    """A simple wrapper to render a nice progress bar with a label."""

    def __init__(self, widgets, min, max):
        self.widgets = widgets
        self._progress = widgets.FloatProgress(min=min, max=max)
        self._label = widgets.Label()
        self._widget = self.widgets.VBox([self._label, self._progress])
        self._displayed = False
        self._disabled = False

    def update(self, value: float, label: str) -> None:
        if self._disabled:
            return
        try:
            self._progress.value = value
            self._label.value = label
            if not self._displayed:
                self._displayed = True
                display_widget(self._widget)
        except Exception as e:
            self._disabled = True
            logger.exception(e)
            wandb.termwarn('Unable to render progress bar, see the user log for details')

    def close(self) -> None:
        if self._disabled or not self._displayed:
            return
        self._widget.close()