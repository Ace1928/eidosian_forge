import abc
from abc import ABC
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero import handlers as H, mc
from minerl.herobraine.hero.mc import ALL_ITEMS, INVERSE_KEYMAP, SIMPLE_KEYBOARD_ACTION
from minerl.herobraine.env_spec import EnvSpec
from typing import List
import numpy as np

        Simple envs have some basic keyboard control functionality, but
        not all.
        