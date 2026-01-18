import os
import sys
import re
def unique_key(adict):
    allkeys = list(adict.keys())
    done = False
    n = 1
    while not done:
        newkey = ''.join([x[:n] for x in allkeys])
        if newkey in allkeys:
            n += 1
        else:
            done = True
    return newkey