from abc import ABC, abstractmethod
from dataclasses import dataclass
from openchat.base import BaseAgent, DecoderLM
def add_bot_message(self, user_id, text):
    self.histories[user_id]['bot_message'].append(text)