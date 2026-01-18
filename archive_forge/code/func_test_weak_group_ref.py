import unittest
import pygame
from pygame import sprite
def test_weak_group_ref(self):
    """
        We create a list of groups, add them to the sprite.
        When we then delete the groups, the sprite should be "dead"
        """
    import gc
    groups = [Group() for Group in self.Groups]
    self.sprite.add(groups)
    del groups
    gc.collect()
    self.assertFalse(self.sprite.alive())