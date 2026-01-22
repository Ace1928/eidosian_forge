import importlib
import logging
class FFVideoOptionalModule(OptionalModule):

    def __init__(self, failMessage=None, require=False):
        super(FFVideoOptionalModule, self).__init__('ffvideo', failMessage=failMessage, require=require)