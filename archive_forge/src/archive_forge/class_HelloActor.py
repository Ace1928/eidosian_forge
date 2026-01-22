from ray.util.client import ray
from typing import Tuple
@ray.remote
class HelloActor:

    def __init__(self):
        self.count = 0

    def say_hello(self, whom: str) -> Tuple[str, int]:
        self.count += 1
        return ('Hello ' + whom, self.count)