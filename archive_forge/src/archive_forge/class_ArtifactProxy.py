import inspect
import pickle
from functools import wraps
from pathlib import Path
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
class ArtifactProxy:

    def __init__(self, flow):
        self.__dict__.update({'flow': flow, 'inputs': {}, 'outputs': {}, 'base': set(dir(flow)), 'params': {p: getattr(flow, p) for p in current.parameter_names}})

    def __setattr__(self, key, val):
        self.outputs[key] = val
        return setattr(self.flow, key, val)

    def __getattr__(self, key):
        if key not in self.base and key not in self.outputs:
            self.inputs[key] = getattr(self.flow, key)
        return getattr(self.flow, key)