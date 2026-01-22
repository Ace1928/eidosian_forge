import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
def _yaml_to_json(self, yaml_templ):
    return yaml.safe_load(yaml_templ)