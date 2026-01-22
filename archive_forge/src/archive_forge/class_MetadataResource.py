import json
from troveclient import base
class MetadataResource(base.Resource):

    def __getitem__(self, item):
        return self.__dict__[item]

    def __contains__(self, item):
        if item in self.__dict__:
            return True
        else:
            return False