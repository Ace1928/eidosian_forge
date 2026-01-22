import os
import pygame
import sys
import tempfile
import time
class SurfaceSubclass(pygame.Surface):
    """A subclassed Surface to test inheritance."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_attribute = True