import sys
from .dependencies import random
def unique_component_name(instance, name):
    if instance.component(name) is None and (not hasattr(instance, name)):
        return name
    name += '_%d' % (randint(0, 9),)
    while True:
        if instance.component(name) is None and (not hasattr(instance, name)):
            return name
        else:
            name += str(randint(0, 9))