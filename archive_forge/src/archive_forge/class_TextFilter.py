from collections import namedtuple
import json
import logging
import pprint
import re
class TextFilter(object):

    @classmethod
    def filter(cls, action_resp_value, filter_params):
        try:
            action, expected_value = filter_params
        except ValueError:
            raise FilterError('Wrong number of filter parameters')
        if action == 'regexp':
            if isinstance(action_resp_value, list):
                resp = list(action_resp_value)
                iterator = enumerate(action_resp_value)
            else:
                resp = dict(action_resp_value)
                iterator = iter(action_resp_value.items())
            remove = []
            for key, value in iterator:
                if not re.search(expected_value, str(value)):
                    remove.append(key)
            if isinstance(resp, list):
                resp = [resp[key] for key, value in enumerate(resp) if key not in remove]
            else:
                resp = dict([(key, value) for key, value in resp.items() if key not in remove])
            return resp
        else:
            raise FilterError('Unknown filter')