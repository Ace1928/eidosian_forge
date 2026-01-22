import unittest
import simplejson as json
class DictForJson(dict):

    def for_json(self):
        return {'alpha': 1}