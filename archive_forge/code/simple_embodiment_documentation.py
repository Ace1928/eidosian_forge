import abc
from abc import ABC
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.mc import INVERSE_KEYMAP
from minerl.herobraine.env_spec import EnvSpec
from typing import List

        Simple envs have some basic keyboard control functionality, but
        not all.
        