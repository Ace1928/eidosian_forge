import pickle
import numpy as np
import shap
from shap._serializable import Deserializer, Serializable, Serializer

        This patched `load` method fix `KernelExplainer.load`.
        Issues in original KernelExplainer.load:
         - Use mismatched model loader to load model
         - Try to load non-existent "masker" attribute
         - Does not load "data" attribute and then cause calling " KernelExplainer"
           constructor lack of "data" argument.
        Note: `model_loader` and `masker_loader` are meaningless argument for
        `KernelExplainer.save`, because the `model` object is saved by pickle dump,
        we must use pickle load to load it.
        and no `masker` for KernelExplainer so `masker_loader` is meaningless.
        but I preserve the 2 argument for overridden API compatibility.
        