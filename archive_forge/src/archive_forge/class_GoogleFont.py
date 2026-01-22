from __future__ import annotations
import json
from typing import Iterable
class GoogleFont(Font):

    def __init__(self, name: str, weights: Iterable[int]=(400, 600)):
        self.name = name
        self.weights = weights

    def stylesheet(self) -> str:
        return f'https://fonts.googleapis.com/css2?family={self.name.replace(' ', '+')}:wght@{';'.join((str(weight) for weight in self.weights))}&display=swap'