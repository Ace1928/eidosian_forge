import copy
import json
import time
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def _tmpl_with_rsrcs(rsrcs, output_value=None):
    tmpl = {'heat_template_version': 'queens', 'parameters': {input_param: {'type': 'string'}}, 'resources': rsrcs}
    if output_value is not None:
        outputs = {'delay_stack': {'value': output_value}}
        tmpl['outputs'] = outputs
    return json.dumps(tmpl)