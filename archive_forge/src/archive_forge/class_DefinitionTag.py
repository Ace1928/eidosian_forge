import os
import zlib
import time  # noqa
import logging
import numpy as np
class DefinitionTag(Tag):
    counter = 0

    def __init__(self):
        Tag.__init__(self)
        DefinitionTag.counter += 1
        self.id = DefinitionTag.counter