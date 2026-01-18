import time
from ase.utils.timing import Timer, timer
@timer('yield')
def yielding(self):
    for i in range(5):
        time.sleep(0.001)
        yield i