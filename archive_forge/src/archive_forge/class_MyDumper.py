import json
import os
import betamax.serializers.base
import yaml
class MyDumper(yaml.Dumper):
    """Specialized Dumper which does nice blocks and unicode."""