from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
class KeyRepeatSimulator:
    """
    Provide simulated repeat key events when given press and
    release events.

    If two or more keys are pressed disable repeating until all
    keys are released.
    """

    def __init__(self, repeat_delay: float, repeat_next: float) -> None:
        """
        repeat_delay -- seconds to wait before starting to repeat keys
        repeat_next -- time between each repeated key
        """
        self.repeat_delay = repeat_delay
        self.repeat_next = repeat_next
        self.pressed: dict[str, float] = {}
        self.multiple_pressed = False

    def press(self, key: str) -> None:
        if self.pressed:
            self.multiple_pressed = True
        self.pressed[key] = time.time()

    def release(self, key: str) -> None:
        if key not in self.pressed:
            return
        del self.pressed[key]
        if not self.pressed:
            self.multiple_pressed = False

    def next_event(self) -> tuple[float, str] | None:
        """
        Return (remaining, key) where remaining is the number of seconds
        (float) until the key repeat event should be sent, or None if no
        events are pending.
        """
        if len(self.pressed) != 1 or self.multiple_pressed:
            return None
        for key, val in self.pressed.items():
            return (max(0.0, val + self.repeat_delay - time.time()), key)
        return None

    def sent_event(self) -> None:
        """
        Cakk this method when you have sent a key repeat event so the
        timer will be reset for the next event
        """
        if len(self.pressed) != 1:
            return
        for key in self.pressed:
            self.pressed[key] = time.time() - self.repeat_delay + self.repeat_next
            return