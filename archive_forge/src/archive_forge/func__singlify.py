from abc import abstractmethod
import types
from minerl.herobraine.hero.handlers.translation import TranslationHandler
import typing
from minerl.herobraine.hero.spaces import Dict
from minerl.herobraine.hero.handler import Handler
from typing import List
import jinja2
import gym
from lxml import etree
import os
import abc
import importlib
from minerl.herobraine.hero import spaces
def _singlify(self, space: spaces.Dict):
    if self.is_single_agent:
        return space.spaces[self.agent_names[0]]
    else:
        return space