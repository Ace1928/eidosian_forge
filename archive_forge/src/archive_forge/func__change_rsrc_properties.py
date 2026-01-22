import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def _change_rsrc_properties(template, rsrcs, values):
    modified_template = copy.deepcopy(template)
    for rsrc_name in rsrcs:
        rsrc_prop = modified_template['resources'][rsrc_name]['properties']
        for prop, new_val in values.items():
            rsrc_prop[prop] = new_val
    return modified_template