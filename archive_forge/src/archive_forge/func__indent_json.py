import json
import os
import betamax.serializers.base
import yaml
def _indent_json(val):
    if not val:
        return ''
    return json.dumps(json.loads(val), indent=2, separators=(',', ': '), sort_keys=False, default=str)