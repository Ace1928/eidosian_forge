import random
import threading
import time
import unittest
from traits.api import Enum, HasStrictTraits
from traits.util.async_trait_wait import wait_for_condition
def test_traits_handler_cleaned_up(self):
    self.lights = TrafficLights(colour='Green')
    t = threading.Thread(target=self.lights.make_random_changes, args=(3,))
    t.start()
    wait_for_condition(condition=lambda l: self.lights.colour == 'Red', obj=self.lights, trait='colour')
    del self.lights
    t.join()